from typing import Optional, List
import os
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from spca_selector_backup import SimpleSPCASelector

class ReservoirFromRolloutCallback(BaseCallback):
    """
    Oblicza wagi SparsePCA dla poszczególnych cech na podstawie top-K komponentów z danych rolloutów.
    Na końcu treningu zapisuje zrzut końcowy (komponenty, wagi per cecha oraz posortowaną listę).
    """

    def __init__(
        self,
        # kadencja rolloutów
        compute_every_rollouts: int = 1,
        warmup_rollouts: int = 2,
        # kadencja raportowania
        print_every_steps: Optional[int] = 100_000,
        print_top_k: int = 20,
        # rezerwuar
        reservoir_size: int = 20_000,
        # konfiguracja SPCA
        n_components: int = 3,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        method: str = "lars",
        weight_norm: str = "l1",           # „l1” (suma |ładunków|) lub „l2” (pierwiastek z sumy kwadratów)
        normalize_weights: bool = True,    # normalizuj wagi tak, by sumowały się do 1
        random_state: int = 0xC0FFEE,
        # I/O / logowanie
        feature_names: Optional[List[str]] = None,
        save_dir='./logs',    # zapisuj tu zrzuty NPZ/CSV
        tensorboard: bool = True,
        verbose: int = 1,
        # kontrola zapisu finałowego
        final_basename: str = "final",     # pliki: spca_final.{npz,csv}, spca_weights_final.csv itd.
    ):
        super().__init__(verbose)
        assert weight_norm in {"l1", "l2"}
        self.compute_every_rollouts = int(compute_every_rollouts)
        self.warmup_rollouts = int(warmup_rollouts)
        self.print_every_steps = int(print_every_steps) if print_every_steps is not None else None
        self.print_top_k = int(print_top_k)
        self.reservoir_size = int(reservoir_size)
        self.n_components = int(max(1, n_components))
        self.alpha = float(alpha)
        self.ridge_alpha = float(ridge_alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.method = str(method)
        self.weight_norm = weight_norm
        self.normalize_weights = bool(normalize_weights)
        self.random_state = int(random_state)
        self.feature_names = feature_names
        self.save_dir = save_dir
        self.tensorboard = tensorboard
        self.final_basename = str(final_basename)

        # stan
        self.rollouts_seen = 0
        self._last_print_ts = 0
        self._rng = np.random.default_rng(self.random_state)
        self._N: Optional[int] = None

        # rezerwuar
        self._X: Optional[np.ndarray] = None   # [R, N]
        self._rows_seen = 0
        self._filled = 0

        # ostatnio wyliczony zrzut (na potrzeby zapisu końcowego)
        self._last_components: Optional[np.ndarray] = None   # [K, N]
        self._last_weights: Optional[np.ndarray] = None      # [N]
        self._last_ranked_idx: Optional[List[int]] = None    # długość N (najlepsze → najgorsze)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    # kompatybilność z SB3
    def _on_step(self) -> bool:
        return True

    # -------------------- funkcje pomocnicze --------------------

    @staticmethod
    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        try:
            if th.is_tensor(x):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def _reservoir_add(self, X_batch: np.ndarray):
        B, N = X_batch.shape
        if self._X is None:
            size = self.reservoir_size if self.reservoir_size > 0 else B
            self._X = np.empty((size, N), dtype=np.float32)
            self._filled = 0
            self._rows_seen = 0
        for i in range(B):
            self._rows_seen += 1
            if self.reservoir_size <= 0:
                if self._filled >= self._X.shape[0]:
                    self._X = np.vstack([self._X, np.empty_like(self._X)])
            if self._filled < self._X.shape[0]:
                dst = self._filled
                self._filled += 1
            else:
                j = self._rng.integers(0, self._rows_seen)
                if j >= self._X.shape[0]:
                    continue
                dst = int(j)
            self._X[dst] = X_batch[i]
    @staticmethod
    def _weights_from_components(comps: np.ndarray, weight_norm: str, normalize: bool) -> np.ndarray:
        if weight_norm == "l1":
            w = np.sum(np.abs(comps), axis=0)
        else:
            w = np.sqrt(np.sum(comps ** 2, axis=0))
        if normalize:
            s = float(np.sum(w))
            if s > 0:
                w = w / s
        return w.astype(np.float64, copy=False)

    @staticmethod
    def _analyze_and_report(self, step_ts: int):
        if self._X is None or self._filled < max(5, self.n_components + 2):
            return

        X = self._X[: self._filled]    # [R, N]
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]

        selector = SimpleSPCASelector(
            feature_names=names,
            n_components=self.n_components,
            alpha=self.alpha,
            ridge_alpha=self.ridge_alpha,
            top_k=self.print_top_k,
        )
        result = selector.compute_from_array(
            X,
            names=names,
            k=self.print_top_k,
            plot=False,
        )
        comps = result["components"]
        weights = self._weights_from_components(
            comps,
            weight_norm=self.weight_norm,
            normalize=self.normalize_weights,
        )
        order = np.argsort(-weights).tolist()

        # zapamiętaj ostatni zrzut do zapisu końcowego
        self._last_components = comps
        self._last_weights = weights
        self._last_ranked_idx = order

        # wypisz podsumowanie
        if self.verbose:
            label = f"step={step_ts:,}"
            print(f"[FeatureSPCA] report @ {label}: reservoir_rows={self._filled:,} (of seen={self._rows_seen:,})  K={comps.shape[0]}  alpha={self.alpha}")
            k = min(self.print_top_k, len(order))
            print(f"  top-{k} features by SPCA weight:")
            for rank, idx in enumerate(order[:k], 1):
                print(f"    {rank:>2}. {names[idx]} (idx={idx})  w={weights[idx]:.6f}")

        # zbiorcze metryki do TensorBoard (kilka skalarów)
        if self.tensorboard:
            try:
                self.logger.record("spca/mean_weight", float(np.mean(weights)))
                self.logger.record("spca/max_weight", float(np.max(weights)))
                self.logger.record("spca/min_weight", float(np.min(weights)))
            except Exception:
                pass

        # periodyczny zrzut
        if self.save_dir:
            stem = f"step_{step_ts:010d}"
            np.savez(
                os.path.join(self.save_dir, f"spca_{stem}.npz"),
                components=comps,               # [K, N]
                weights=weights,                # [N]
                feature_names=np.array(names),
                n_components=comps.shape[0],
                alpha=self.alpha,
                ridge_alpha=self.ridge_alpha,
                rows_filled=self._filled,
                rows_seen=self._rows_seen,
            )
            try:
                import pandas as pd
                pd.DataFrame({"feature": names, "weight": weights}).to_csv(
                    os.path.join(self.save_dir, f"spca_weights_{stem}.csv"), index=False
                )
                pd.DataFrame(comps, columns=names).to_csv(
                    os.path.join(self.save_dir, f"spca_components_{stem}.csv"), index=False
                )
                # lista posortowana
                pd.DataFrame({
                    "rank": np.arange(1, len(order)+1),
                    "idx": order,
                    "feature": [names[i] for i in order],
                    "weight": [weights[i] for i in order],
                }).to_csv(os.path.join(self.save_dir, f"spca_ranking_{stem}.csv"), index=False)
            except Exception:
                pass

    # ---------------- haki SB3 ----------------

    def _on_rollout_end(self) -> bool:
        self.rollouts_seen += 1
        if self.rollouts_seen < self.warmup_rollouts:
            return True
        if (self.rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
            return True

        buf = self.model.rollout_buffer
        obs = self._to_np(buf.observations)  # [T, B, N] dla (Recurrent)PPO
        if obs.ndim == 2:                    # rzadki fallback: [T, N] -> [T,1,N]
            obs = obs[:, None, :]
        T, B, N = obs.shape
        if self._N is None:
            self._N = N
            if self.feature_names is not None and len(self.feature_names) != N:
                raise ValueError(f"feature_names length {len(self.feature_names)} != N={N}")

        X_rows = obs.reshape(-1, N).astype(np.float32)

        # dodaj do rezerwuaru
        if self.reservoir_size == 0:
            self._X = X_rows.copy()
            self._filled = len(X_rows)
            self._rows_seen += len(X_rows)
        else:
            self._reservoir_add(X_rows)

        # kadencja logowania zsynchronizowana z końcem rolloutów
        if self.print_every_steps is not None:
            if (self.num_timesteps - self._last_print_ts) >= self.print_every_steps:
                self._analyze_and_report(self.num_timesteps)
                self._last_print_ts = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        """
        Zapisz końcowy zrzut komponentów, wag i rankingu.
        Korzysta z bieżącego rezerwuaru; przed zapisem ponownie liczy statystyki, by mieć najnowsze dane.
        """
        if self._X is None or self._filled < max(5, self.n_components + 2):
            if self.verbose:
                print("[FeatureSPCA] training_end: not enough data to save final snapshot.")
            return

        # upewnij się, że snapshot jest aktualny
        self._analyze_and_report(self.num_timesteps)

        if not self.save_dir:
            if self.verbose:
                print("[FeatureSPCA] training_end: save_dir is None, skipping file save.")
            return

        names = self.feature_names or [f"f{i}" for i in range(self._X.shape[1])]
        comps = self._last_components
        weights = self._last_weights
        order = self._last_ranked_idx or list(np.argsort(-weights).tolist())

        stem = self.final_basename  # np. „final”
        # pakiet NPZ
        np.savez(
            os.path.join(self.save_dir, f"spca_{stem}.npz"),
            components=comps,               # [K, N]
            weights=weights,                # [N]
            ranked_indices=np.array(order, dtype=np.int64),
            feature_names=np.array(names),
            n_components=comps.shape[0] if comps is not None else 0,
            alpha=self.alpha,
            ridge_alpha=self.ridge_alpha,
            rows_filled=self._filled,
            rows_seen=self._rows_seen,
            step=self.num_timesteps,
        )
        # pliki CSV
        try:
            import pandas as pd
            pd.DataFrame({"feature": names, "weight": weights}).to_csv(
                os.path.join(self.save_dir, f"spca_weights_{stem}.csv"), index=False
            )
            if comps is not None:
                pd.DataFrame(comps, columns=names).to_csv(
                    os.path.join(self.save_dir, f"spca_components_{stem}.csv"), index=False
                )
            pd.DataFrame({
                "rank": np.arange(1, len(order)+1),
                "idx": order,
                "feature": [names[i] for i in order],
                "weight": [weights[i] for i in order],
            }).to_csv(os.path.join(self.save_dir, f"spca_ranking_{stem}.csv"), index=False)
        except Exception:
            pass

        # po zapisaniu plików spca_final...
        if self.save_dir and self._X is not None and self._filled > 0:
            names = self.feature_names or [f"f{i}" for i in range(self._X.shape[1])]
            np.savez(
                os.path.join(self.save_dir, "spca_reservoir_final.npz"),
                X=self._X[: self._filled],                  # [R, N]
                feature_names=np.array(names),
                rows_filled=self._filled,
                rows_seen=self._rows_seen,
                step=self.num_timesteps,
            )
            if self.verbose:
                print(f"[FeatureSPCA] saved reservoir to {os.path.join(self.save_dir, 'spca_reservoir_final.npz')}")


        if self.verbose:
            print(f"[FeatureSPCA] saved final SPCA snapshot to {self.save_dir} (basename='{stem}')")


import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

# -- pomocnicze: odrzucanie skorelowanych cech na podstawie rezerwuaru --
def correlation_filter(X, ranked_idx, keep_k, corr_threshold=0.95):
    X = np.asarray(X)
    N = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)

    selected = []
    for i in ranked_idx:
        if selected and np.any(np.abs(C[i, selected]) >= corr_threshold):
            continue
        selected.append(i)
        if len(selected) >= keep_k:
            break
    if len(selected) < keep_k:
        for i in ranked_idx:
            if i not in selected:
                selected.append(i)
                if len(selected) >= keep_k:
                    break
    return selected[:keep_k]


class AttentionFromRolloutCallback_BACKUP(BaseCallback):
    """
    Kolektor uwag kompatybilny z RecurrentPPO/PPO, agregujący CAŁY dotychczasowy trening.

    Zbiera OBU warianty:
      - metric_importance (tylko uwaga)
      - contrib_importance (uwaga ważona wkładem)

    Skumulowane statystyki epizodowe dla każdej odmiany:
      • cum_mean_* [N] : bieżąca średnia Welforda dla wektorów epizodowych
      • cum_var_*  [N] : bieżąca wariancja Welforda (M2)
      • freq_top_m_*[N]: licznik pojawień w TOP-M dla epizodu
      • ep_count       : łączna liczba zebranych epizodów

    Możesz wybrać, który wariant służy do drukowania/rankingu/masek:
      - rank_source: "contrib" (domyślnie) | "metric"
      - mask_source: "contrib" (domyślnie) | "metric"
    """
    def __init__(
        self,
        compute_every_rollouts: int = 1,
        warmup_rollouts: int = 2,
        # drukowanie/logowanie
        print_every_steps=100_000,
        print_top_k: int = 15,
        top_m_for_frequency=None,  # domyślnie ustawiane na 2*sqrt(N)
        feature_names=None,
        # przycinanie cech (opcjonalne)
        select_k=None,
        apply_mask: bool = False,
        corr_threshold: float = 0.95,
        # rezerwuar na potrzeby korelacji (zachowuje surowe wiersze z perspektywy polityki)
        reservoir_size: int = 20000,
        # wariant używany do drukowania/rankingu/masek
        rank_source: str = "contrib",  # „contrib” | „metric”
        mask_source: str = "contrib",  # „contrib” | „metric”
        # zapisywanie/logowanie
        save_npz_path: str = "logs/attn_cumulative/attn_cumulative_final.npz",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.compute_every_rollouts = int(compute_every_rollouts)
        self.warmup_rollouts = int(warmup_rollouts)

        self.print_every_steps = int(print_every_steps) if print_every_steps is not None else None
        self.print_top_k = int(print_top_k)
        self.top_m_for_frequency = top_m_for_frequency
        self.feature_names = feature_names

        self.select_k = select_k
        self.apply_mask = bool(apply_mask)
        self.corr_threshold = float(corr_threshold)

        # wybór wariantu
        self.rank_source = str(rank_source).lower()
        self.mask_source = str(mask_source).lower()
        assert self.rank_source in {"contrib", "metric"}
        assert self.mask_source in {"contrib", "metric"}

        # statystyki skumulowane (wspólny licznik epizodów)
        self.ep_count = 0
        # wariant tylko metric (wagi uwagi)
        self.cum_mean_metric = None
        self.cum_M2_metric = None
        self.freq_top_m_metric = None
        # wariant z uwzględnieniem wkładu
        self.cum_mean_contrib = None
        self.cum_M2_contrib = None
        self.freq_top_m_contrib = None

        # kadencja drukowania
        self._last_print_ts = 0

        # rezerwuar na korelację
        self.reservoir_size = int(reservoir_size)
        self._reservoir = None
        self._reservoir_filled = 0
        self._rows_seen = 0

        self.rollouts_seen = 0
        self.save_npz_path = save_npz_path
        self._cum_snapshots = []  # lista słowników

    # -------- narzędzia --------
    def _reservoir_add(self, X_batch):
        if self.reservoir_size <= 0:
            return
        X_batch = np.asarray(X_batch)
        if X_batch.ndim == 1:
            X_batch = X_batch[None, :]
        N = X_batch.shape[1]
        if self._reservoir is None:
            self._reservoir = np.empty((self.reservoir_size, N), dtype=X_batch.dtype)
            self._reservoir_filled = 0
            self._rows_seen = 0

        for row in X_batch:
            self._rows_seen += 1
            if self._reservoir_filled < self.reservoir_size:
                self._reservoir[self._reservoir_filled] = row
                self._reservoir_filled += 1
            else:
                j = np.random.randint(0, self._rows_seen)
                if j < self.reservoir_size:
                    self._reservoir[j] = row

    def _pretty_print_ranking(self, ranked, mean_vec, freq_vec, var_vec, step_ts, tag):
        names = self.feature_names or [f"f{i}" for i in range(len(mean_vec))]
        k = min(self.print_top_k, len(ranked))
        print(f"[AttentionFromRollout:{tag}] step={step_ts:,}  cumulative top-{k} features")
        for r, i in enumerate(ranked[:k], 1):
            f = names[i]
            print(f"  {r:>2}. {f} (idx={i})  freq={freq_vec[i]:.3f}  mean={mean_vec[i]:.6f}  var={var_vec[i]:.6f}")

    # -------- haki SB3 --------
    def _on_rollout_end_bp(self) -> bool:
        """
        Przy każdym rolloucie zawsze dodaj obserwacje do rezerwuaru.
        Następnie, po rozgrzewce i spełnieniu kadencji, opcjonalnie uruchom analizę SPCA.
        """
        self.rollouts_seen += 1

        # ---- 1) pobierz obserwacje rolloutu i zaktualizuj rezerwuar ----
        buf = self.model.rollout_buffer
        obs = self._to_np(buf.observations)  # PPO / RecurrentPPO: [T, B, N]
        if obs.ndim == 2:                     # rzadki fallback: [T, N] -> [T,1,N]
            obs = obs[:, None, :]
        T, B, N = obs.shape

        if self._N is None:
            self._N = N
            if self.feature_names is not None and len(self.feature_names) != N:
                raise ValueError(
                    f"feature_names length {len(self.feature_names)} != N={N}"
                )

        X_rows = obs.reshape(-1, N).astype(np.float32)  # [T*B, N]

        # zawsze dodawaj do rezerwuaru
        if self.reservoir_size == 0:
            # przypadek szczególny: przechowuj wszystkie wiersze
            self._X = X_rows.copy()
            self._filled = len(X_rows)
            self._rows_seen += len(X_rows)
        else:
            self._reservoir_add(X_rows)

        # ---- 2) rozgrzewka / kadencja tylko dla obliczeń SPCA ----
        if self.rollouts_seen < self.warmup_rollouts:
            return True
        if (self.rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
            return True

        # ---- 3) opcjonalnie analizuj i zapisuj zrzut pośredni ----
        if self.print_every_steps is not None:
            if (self.num_timesteps - self._last_print_ts) >= self.print_every_steps:
                self._analyze_and_report(self.num_timesteps)
                self._last_print_ts = self.num_timesteps

        return True

    def _on_rollout_end(self) -> bool:
        self.rollouts_seen += 1
        if self.rollouts_seen < self.warmup_rollouts:
            return True
        if (self.rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
            return True

        buf = self.model.rollout_buffer
        obs = buf.observations                 # [T, n_envs, N]
        ep_starts = buf.episode_starts         # [T, n_envs]
        if obs.ndim == 2:  # awaryjnie, gdy bufor został spłaszczony
            obs = obs[:, None, :]
            ep_starts = ep_starts[:, None]

        T, B, N = obs.shape
        device = self.model.policy.device
        fe = self.model.policy.features_extractor

        # jedno przejście w przód, aby wypełnić diagnostykę uwagi
        was_training = fe.training
        fe.eval()
        with th.no_grad():
            x = th.as_tensor(obs, device=device, dtype=th.float32)  # [T,B,N]
            _ = fe(x)  # uruchamia ekstraktor; wypełnia .metric_importance / .contrib_importance
            v_metric = getattr(fe, "metric_importance", None)
            v_contrib = getattr(fe, "contrib_importance", None)

            metric_ok = (v_metric is not None)
            contrib_ok = (v_contrib is not None)

            if metric_ok:
                v_metric = v_metric.detach().cpu().numpy().reshape(T, B, N)
            if contrib_ok:
                v_contrib = v_contrib.detach().cpu().numpy().reshape(T, B, N)
        if was_training:
            fe.train()

        # zbierz średnie per-epizod dla każdej odmiany
        per_ep_metric = []
        per_ep_contrib = []
        for b in range(B):
            start = 0
            for t in range(T):
                if ep_starts[t, b]:
                    if t > start:
                        if metric_ok:
                            per_ep_metric.append(v_metric[start:t, b, :].mean(axis=0))
                        if contrib_ok:
                            per_ep_contrib.append(v_contrib[start:t, b, :].mean(axis=0))
                    start = t
            if start < T:
                if metric_ok:
                    per_ep_metric.append(v_metric[start:T, b, :].mean(axis=0))
                if contrib_ok:
                    per_ep_contrib.append(v_contrib[start:T, b, :].mean(axis=0))

        got_any = (len(per_ep_metric) > 0) or (len(per_ep_contrib) > 0)
        if not got_any:
            return True

        if metric_ok and len(per_ep_metric) > 0:
            per_ep_metric = np.stack(per_ep_metric, axis=0)  # [E_r, N]
        else:
            per_ep_metric = None

        if contrib_ok and len(per_ep_contrib) > 0:
            per_ep_contrib = np.stack(per_ep_contrib, axis=0)  # [E_r, N]
        else:
            per_ep_contrib = None

        # ---- aktualizacja skumulowanych statystyk Welforda ----
        # inicjalizuj tablice leniwie przy pierwszym dostępnym wariancie
        def _ensure_init(shapeN):
            if self.cum_mean_metric is None and metric_ok:
                self.cum_mean_metric = np.zeros(shapeN, dtype=np.float32)
                self.cum_M2_metric = np.zeros(shapeN, dtype=np.float32)
                self.freq_top_m_metric = np.zeros(shapeN, dtype=np.float32)
            if self.cum_mean_contrib is None and contrib_ok:
                self.cum_mean_contrib = np.zeros(shapeN, dtype=np.float32)
                self.cum_M2_contrib = np.zeros(shapeN, dtype=np.float32)
                self.freq_top_m_contrib = np.zeros(shapeN, dtype=np.float32)

        _ensure_init(N)

        # licznik epizodów zwiększa się o liczbę epizodów w danym rolloucie
        # wykorzystaj tę tablicę epizodową, która nie jest None
        E_r = 0
        if per_ep_metric is not None:
            E_r = per_ep_metric.shape[0]
        elif per_ep_contrib is not None:
            E_r = per_ep_contrib.shape[0]
        self.ep_count += int(E_r)

        # zwektoryzowana aktualizacja Welforda dla każdego epizodu
        if per_ep_metric is not None:
            for v in per_ep_metric:
                delta = v - self.cum_mean_metric
                self.cum_mean_metric += delta / self.ep_count
                delta2 = v - self.cum_mean_metric
                self.cum_M2_metric += delta * delta2

        if per_ep_contrib is not None:
            for v in per_ep_contrib:
                delta = v - self.cum_mean_contrib
                self.cum_mean_contrib += delta / self.ep_count
                delta2 = v - self.cum_mean_contrib
                self.cum_M2_contrib += delta * delta2

        # aktualizacja częstości TOP-M
        m = int(self.top_m_for_frequency or max(1, min(N, 2 * int(np.ceil(np.sqrt(N))))))
        if per_ep_metric is not None:
            topm_idx = np.argsort(-per_ep_metric, axis=1)[:, :m]
            for row in topm_idx:
                self.freq_top_m_metric[row] += 1.0
        if per_ep_contrib is not None:
            topm_idx = np.argsort(-per_ep_contrib, axis=1)[:, :m]
            for row in topm_idx:
                self.freq_top_m_contrib[row] += 1.0

        # utrzymuj rezerwuar do odrzucania skorelowanych cech (wiersze w formie widzianej przez politykę)
        self._reservoir_add(obs.reshape(-1, N))

        # ---- logowanie do TensorBoard ----
        # wariant metric
        for i, val in enumerate(self.cum_mean_metric):
            self.logger.record(f"attn/mean_f{i}/M", float(val))
        self.logger.record("attn/episodes/M", int(self.ep_count))
        self.logger.record("attn/mean_all/M", float(self.cum_mean_metric.mean()))

        # wariant contrib
        for i, val in enumerate(self.cum_mean_contrib):
            self.logger.record(f"attn/mean_f{i}/C", float(val))
        self.logger.record("attn/episodes/C", int(self.ep_count))
        self.logger.record("attn/mean_all/C", float(self.cum_mean_contrib.mean()))


        # ---- opcjonalnie: nałóż maskę, używając skumulowanego rankingu i korelacji z rezerwuarem ----
        if (self.select_k is not None) and self.apply_mask:
            # wybierz źródło dla maski
            if self.mask_source == "contrib" and self.cum_mean_contrib is not None:
                freq = self.freq_top_m_contrib / max(1, self.ep_count)
                var  = (self.cum_M2_contrib / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_contrib)
                base = self.cum_mean_contrib
            else:
                freq = self.freq_top_m_metric / max(1, self.ep_count)
                var  = (self.cum_M2_metric / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_metric)
                base = self.cum_mean_metric

            order = np.lexsort((var, -base, -freq))   # najlepsze -> najgorsze
            ranked = order.tolist()

            X_corr = (self._reservoir[:self._reservoir_filled]
                      if (self._reservoir is not None and self._reservoir_filled > 1)
                      else obs.reshape(-1, N))
            selected = correlation_filter(X_corr, ranked, keep_k=self.select_k, corr_threshold=self.corr_threshold)

            mask = np.zeros(N, dtype=np.float32); mask[selected] = 1.0
            if hasattr(fe, "set_active_mask"):
                fe.set_active_mask(mask)
                if self.verbose:
                    names = self.feature_names or [f"f{i}" for i in range(N)]
                    picked = [names[i] for i in selected]
                    print(f"[AttentionFromRollout:{self.mask_source}] applied cumulative mask top-{self.select_k}: {picked}")

        # ---- skumulowany wydruk co n kroków (zsynchronizowany z końcem rolloutu) ----
        if self.print_every_steps is not None and (self.num_timesteps - self._last_print_ts) >= self.print_every_steps:
            if self.rank_source == "contrib" and self.cum_mean_contrib is not None:
                freq = self.freq_top_m_contrib / max(1, self.ep_count)
                var  = (self.cum_M2_contrib / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_contrib)
                base = self.cum_mean_contrib
                order = np.lexsort((var, -base, -freq))
                self._pretty_print_ranking(order.tolist(), base, freq, var, self.num_timesteps, tag="contrib")
            elif self.cum_mean_metric is not None:
                freq = self.freq_top_m_metric / max(1, self.ep_count)
                var  = (self.cum_M2_metric / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_metric)
                base = self.cum_mean_metric
                order = np.lexsort((var, -base, -freq))
                self._pretty_print_ranking(order.tolist(), base, freq, var, self.num_timesteps, tag="metric")
            self._last_print_ts = self.num_timesteps

        # zrzut do zapisu
        snap = {"step": int(self.num_timesteps)}
        if self.cum_mean_metric is not None:
            snap.update({
                "mean_metric": self.cum_mean_metric.copy(),
                "var_metric":  (self.cum_M2_metric / max(1, self.ep_count - 1)).copy() if self.ep_count > 1 else np.zeros_like(self.cum_mean_metric),
                "freq_metric": (self.freq_top_m_metric / max(1, self.ep_count)).copy(),
            })
        if self.cum_mean_contrib is not None:
            snap.update({
                "mean_contrib": self.cum_mean_contrib.copy(),
                "var_contrib":  (self.cum_M2_contrib / max(1, self.ep_count - 1)).copy() if self.ep_count > 1 else np.zeros_like(self.cum_mean_contrib),
                "freq_contrib": (self.freq_top_m_contrib / max(1, self.ep_count)).copy(),
            })
        self._cum_snapshots.append(snap)

        return True

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if not self._cum_snapshots or self.save_npz_path is None:
            return
        # skonsoliduj zrzuty (brakujące klucze wypełnij zerami, żeby zapis był spójny)
        steps = np.array([s["step"] for s in self._cum_snapshots], dtype=np.int64)

        def _stack_or_zeros(key, N):
            arrs = []
            for s in self._cum_snapshots:
                if key in s:
                    arrs.append(s[key])
                else:
                    arrs.append(np.zeros(N, dtype=np.float32))
            return np.stack(arrs, axis=0)

        # wyprowadź N z dowolnej dostępnej średniej
        if self.cum_mean_contrib is not None:
            N = len(self.cum_mean_contrib)
        elif self.cum_mean_metric is not None:
            N = len(self.cum_mean_metric)
        else:
            N = 0

        means_metric = _stack_or_zeros("mean_metric", N)
        vars_metric  = _stack_or_zeros("var_metric", N)
        freqs_metric = _stack_or_zeros("freq_metric", N)

        means_contrib = _stack_or_zeros("mean_contrib", N)
        vars_contrib  = _stack_or_zeros("var_contrib", N)
        freqs_contrib = _stack_or_zeros("freq_contrib", N)

        names = np.array(self.feature_names or [f"f{i}" for i in range(N)])

        np.savez(
            self.save_npz_path,
            steps=steps,
            # serie wyłącznie dla wariantu metric
            means_metric=means_metric, vars_metric=vars_metric, freqs_metric=freqs_metric,
            # serie dla wariantu contribution-aware
            means_contrib=means_contrib, vars_contrib=vars_contrib, freqs_contrib=freqs_contrib,
            feature_names=names,
            ep_count=np.array([self.ep_count], dtype=np.int64),
            rows_seen=np.array([self._rows_seen], dtype=np.int64),
            reservoir_filled=np.array([self._reservoir_filled], dtype=np.int64),
            rank_source=np.array([self.rank_source]),
            mask_source=np.array([self.mask_source]),
        )
        if self.verbose:
            print(f"[AttentionFromRollout] saved cumulative stats (metric & contrib) to {self.save_npz_path}")

