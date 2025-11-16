from typing import Optional, List
import os
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




from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pickle
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class EpisodeRollout:
    """
    Pojedynczy rollout (pełny epizod) z jednej instancji środowiska.
    """
    obs: List[np.ndarray]
    next_obs: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]

    def to_numpy_dict(self) -> Dict[str, Any]:
        """Konwersja na słownik z tablicami numpy (łatwiejszy późniejszy użytek)."""
        return {
            "obs": np.stack(self.obs) if self.obs else np.zeros((0,)),
            "next_obs": np.stack(self.next_obs) if self.next_obs else np.zeros((0,)),
            "actions": np.stack(self.actions) if self.actions else np.zeros((0,)),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=bool),
            "infos": list(self.infos),
        }


class RollingRolloutBufferCallback(BaseCallback):
    """
    Callback zbierający ostatnie `buffer_size` epizodów (rolloutów) w trakcie treningu.

    - Działa z VecEnv.
    - Każdy zapisany element to pełny epizod jednej instancji środowiska od resetu do done.
    - Utrzymuje przesuwające się okno (deque) maksymalnie `buffer_size` epizodów.
    - Opcjonalnie zapisuje bufor na dysk po zakończeniu treningu.
    """

    def __init__(
        self,
        buffer_size: int = 100,
        save_path: Optional[str] = None,
        save_on_training_end: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.buffer_size = int(buffer_size)
        self.rollouts: Deque[Dict[str, Any]] = deque(maxlen=self.buffer_size)
        self._current_episodes: List[EpisodeRollout] = []

        self.save_path = save_path
        self.save_on_training_end = bool(save_on_training_end)

    def _on_training_start(self) -> None:
        # zainicjalizuj bufory epizodów dla każdej instancji środowiska
        n_envs = self.training_env.num_envs
        self._current_episodes = [
            EpisodeRollout(obs=[], next_obs=[], actions=[], rewards=[], dones=[], infos=[])
            for _ in range(n_envs)
        ]

    def _on_step(self) -> bool:
        """
        Wywoływany po każdym env.step() wewnątrz collect_rollouts().
        Korzysta z lokalnych zmiennych algorytmu:
          - new_obs, rewards, dones, infos, actions
        oraz poprzedniej obserwacji z model._last_obs.
        """
        new_obs = self.locals["new_obs"]      # (n_envs, *obs_shape)
        rewards = self.locals["rewards"]      # (n_envs,)
        dones = self.locals["dones"]          # (n_envs,)
        infos = self.locals["infos"]          # lista długości n_envs
        actions = self.locals["actions"]      # (n_envs, ...) dyskretne lub ciągłe
        last_obs = self.model._last_obs       # obserwacje użyte do wyznaczenia akcji

        n_envs = len(dones)
        for env_idx in range(n_envs):
            ep = self._current_episodes[env_idx]

            o_t = np.array(last_obs[env_idx], copy=True)
            o_tp1 = np.array(new_obs[env_idx], copy=True)
            a_t = np.array(actions[env_idx], copy=True)
            r_t = float(rewards[env_idx])
            d_t = bool(dones[env_idx])
            info_t = infos[env_idx]

            ep.obs.append(o_t)
            ep.next_obs.append(o_tp1)
            ep.actions.append(a_t)
            ep.rewards.append(r_t)
            ep.dones.append(d_t)
            ep.infos.append(info_t)

            if d_t:
                # epizod zakończony -> dodaj do bufora cyklicznego
                self.rollouts.append(ep.to_numpy_dict())
                # rozpocznij nowy bufor epizodu dla tego środowiska
                self._current_episodes[env_idx] = EpisodeRollout(
                    obs=[], next_obs=[], actions=[], rewards=[], dones=[], infos=[]
                )

        return True  # kontynuuj trening

    def get_rollouts(self) -> List[Dict[str, Any]]:
        """Zwróć wszystkie zebrane rollouty jako listę."""
        return list(self.rollouts)

    def save_buffer(self, path: Optional[str] = None) -> None:
        """
        Zapisz bieżący bufor na dysk przy użyciu pickle.
        """
        save_path = path or self.save_path
        if save_path is None:
            if self.verbose > 0:
                print("[RollingRolloutBufferCallback] No save_path specified, skipping save.")
            return

        data = self.get_rollouts()
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        if self.verbose > 0:
            print(f"[RollingRolloutBufferCallback] zapisano {len(data)} rolloutów do {save_path}")

    def _on_training_end(self) -> None:
        """
        Wywoływany na samym końcu treningu.
        W razie potrzeby zapisuje bufor na dysk.
        """
        if self.save_on_training_end:
            self.save_buffer()


import numpy as np
import torch as th
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


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


class AttentionFromRolloutCallback(BaseCallback):
    """
    Attention callback, który:

      • liczy skumulowane statystyki uwagi (metric & contrib) per-epizod,
      • trzyma rolling buffer PEŁNYCH epizodów (od resetu do done) z wszystkich envów,
      • do odrzucania skorelowanych cech używa korelacji liczonej na ostatnich epizodach.

    Parametr `reservoir_size` oznacza teraz MAKS. LICZBĘ EPIZODÓW w buforze,
    analogicznie do `buffer_size` w RollingRolloutBufferCallback.
    """

    def __init__(
        self,
        compute_every_rollouts: int = 1,
        warmup_rollouts: int = 2,
        # drukowanie/logowanie
        print_every_steps=100_000,
        print_top_k: int = 15,
        top_m_for_frequency=None,
        feature_names=None,
        # przycinanie cech (opcjonalne)
        select_k=None,
        apply_mask: bool = False,
        corr_threshold: float = 0.95,
        # teraz: maksymalna liczba PEŁNYCH epizodów w buforze
        reservoir_size: int = 200,
        # wariant używany do drukowania/rankingu/masek
        rank_source: str = "contrib",
        mask_source: str = "contrib",
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

        self.rank_source = str(rank_source).lower()
        self.mask_source = str(mask_source).lower()
        assert self.rank_source in {"contrib", "metric"}
        assert self.mask_source in {"contrib", "metric"}

        # skumulowane statystyki
        self.ep_count = 0
        # metric-only
        self.cum_mean_metric = None
        self.cum_M2_metric = None
        self.freq_top_m_metric = None
        # contribution-aware
        self.cum_mean_contrib = None
        self.cum_M2_contrib = None
        self.freq_top_m_contrib = None

        self._last_print_ts = 0

        # rolling buffer PEŁNYCH epizodów (pełne rollouty z jednego env)
        # reservoir_size = maksymalna liczba epizodów w buforze
        self.reservoir_size = int(reservoir_size)
        self._episode_obs_buffer = deque(maxlen=self.reservoir_size) if self.reservoir_size > 0 else None
        self._current_ep_obs = None    # list[list[obs]], inicjalizacja w _on_training_start
        self._rows_seen = 0            # łączna liczba wierszy wrzuconych do bufora (dla statystyk)
        self._reservoir_filled = 0     # ile epizodów faktycznie siedzi w buforze

        self.rollouts_seen = 0
        self.save_npz_path = save_npz_path
        self._cum_snapshots = []

    # -------- narzędzia --------
    def _store_episode_obs(self, env_idx: int):
        """
        Zapisz zakończony epizod dla danego środowiska do bufora epizodów.
        Korzysta z self._current_ep_obs[env_idx], która jest listą obserwacji.
        """
        if self._current_ep_obs is None:
            return

        obs_list = self._current_ep_obs[env_idx]
        if not obs_list:
            return

        ep_arr = np.stack(obs_list, axis=0).astype(np.float32)  # [L, N]
        self._rows_seen += ep_arr.shape[0]

        if self._episode_obs_buffer is not None:
            self._episode_obs_buffer.append(ep_arr)
            self._reservoir_filled = min(self._reservoir_filled + 1, self.reservoir_size)

        # wyczyść bufor bieżącego epizodu
        self._current_ep_obs[env_idx] = []

    def _pretty_print_ranking(self, ranked, mean_vec, freq_vec, var_vec, step_ts, tag):
        names = self.feature_names or [f"f{i}" for i in range(len(mean_vec))]
        k = min(self.print_top_k, len(ranked))
        print(f"[AttentionFromRollout:{tag}] step={step_ts:,}  cumulative top-{k} features")
        for r, i in enumerate(ranked[:k], 1):
            f = names[i]
            print(f"  {r:>2}. {f} (idx={i})  freq={freq_vec[i]:.3f}  mean={mean_vec[i]:.6f}  var={var_vec[i]:.6f}")

    # -------- haki SB3 --------
    def _on_training_start(self) -> None:
        # inicjalizacja buforów epizodów (po jednym na każde env)
        if self.training_env is None:
            return
        n_envs = self.training_env.num_envs
        self._current_ep_obs = [[] for _ in range(n_envs)]

    def _on_step(self) -> bool:
        """
        Zbieranie PEŁNYCH epizodów (rolloutów) przy pomocy sygnałów z collect_rollouts.

        Na każdym kroku zapisujemy obserwację, na podstawie której była liczona akcja
        (model._last_obs). Gdy env zgłasza done, zamykamy epizod i wrzucamy go
        do bufora epizodów self._episode_obs_buffer (rolling window).
        """
        if self._current_ep_obs is None:
            return True

        new_obs = self.locals["new_obs"]   # nieużywane, ale dostępne
        dones = self.locals["dones"]
        last_obs = self.model._last_obs    # obserwacja, na której policzono akcję

        n_envs = len(dones)
        for env_idx in range(n_envs):
            o_t = np.array(last_obs[env_idx], copy=True)
            self._current_ep_obs[env_idx].append(o_t)

            if bool(dones[env_idx]):
                # pełny epizod (od reset do done) -> dodaj do rolling buffera
                self._store_episode_obs(env_idx)

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
            _ = fe(x)  # wypełnia fe.metric_importance / fe.contrib_importance
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

        # ---- średnie per-epizod dla każdej odmiany ----
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
        E_r = 0
        if per_ep_metric is not None:
            E_r = per_ep_metric.shape[0]
        elif per_ep_contrib is not None:
            E_r = per_ep_contrib.shape[0]
        self.ep_count += int(E_r)

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

        # ---- aktualizacja częstości TOP-M ----
        m = int(self.top_m_for_frequency or max(1, min(N, 2 * int(np.ceil(np.sqrt(N))))))
        if per_ep_metric is not None:
            topm_idx = np.argsort(-per_ep_metric, axis=1)[:, :m]
            for row in topm_idx:
                self.freq_top_m_metric[row] += 1.0
        if per_ep_contrib is not None:
            topm_idx = np.argsort(-per_ep_contrib, axis=1)[:, :m]
            for row in topm_idx:
                self.freq_top_m_contrib[row] += 1.0

        # ---- logowanie do TensorBoard ----
        if self.cum_mean_metric is not None:
            for i, val in enumerate(self.cum_mean_metric):
                self.logger.record(f"attn/mean_f{i}/M", float(val))
            self.logger.record("attn/episodes/M", int(self.ep_count))
            self.logger.record("attn/mean_all/M", float(self.cum_mean_metric.mean()))

        if self.cum_mean_contrib is not None:
            for i, val in enumerate(self.cum_mean_contrib):
                self.logger.record(f"attn/mean_f{i}/C", float(val))
            self.logger.record("attn/episodes/C", int(self.ep_count))
            self.logger.record("attn/mean_all/C", float(self.cum_mean_contrib.mean()))

        # ---- opcjonalnie: nałóż maskę, używając skumulowanego rankingu i PEŁNYCH epizodów ----
        if (self.select_k is not None) and self.apply_mask:
            if self.mask_source == "contrib" and self.cum_mean_contrib is not None:
                freq = self.freq_top_m_contrib / max(1, self.ep_count)
                var  = (self.cum_M2_contrib / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_contrib)
                base = self.cum_mean_contrib
            else:
                freq = self.freq_top_m_metric / max(1, self.ep_count)
                var  = (self.cum_M2_metric / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_metric)
                base = self.cum_mean_metric

            order = np.lexsort((var, -base, -freq))
            ranked = order.tolist()

            # dane do korelacji:
            # - jeśli mamy bufor PEŁNYCH epizodów, wykorzystujemy wszystkie ich wiersze
            # - w przeciwnym razie fallback na aktualny rollout
            if self._episode_obs_buffer and len(self._episode_obs_buffer) > 0:
                X_corr = np.concatenate(list(self._episode_obs_buffer), axis=0)  # [sum_L, N]
            else:
                X_corr = obs.reshape(-1, N)

            selected = correlation_filter(
                X_corr,
                ranked,
                keep_k=self.select_k,
                corr_threshold=self.corr_threshold,
            )

            mask = np.zeros(N, dtype=np.float32)
            mask[selected] = 1.0
            if hasattr(fe, "set_active_mask"):
                fe.set_active_mask(mask)
                if self.verbose:
                    names = self.feature_names or [f"f{i}" for i in range(N)]
                    picked = [names[i] for i in selected]
                    print(f"[AttentionFromRollout:{self.mask_source}] applied cumulative mask top-{self.select_k} using FULL episodes: {picked}")

        # ---- skumulowany wydruk co n kroków ----
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

    def _on_training_end(self) -> None:
        if not self._cum_snapshots or self.save_npz_path is None:
            return

        steps = np.array([s["step"] for s in self._cum_snapshots], dtype=np.int64)

        def _stack_or_zeros(key, N):
            arrs = []
            for s in self._cum_snapshots:
                if key in s:
                    arrs.append(s[key])
                else:
                    arrs.append(np.zeros(N, dtype=np.float32))
            return np.stack(arrs, axis=0)

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
            means_metric=means_metric, vars_metric=vars_metric, freqs_metric=freqs_metric,
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
