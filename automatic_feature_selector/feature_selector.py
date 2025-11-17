

import os
from typing import Any, Dict, Optional, Sequence, Tuple, List
import pickle

import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import SparsePCA
import stable_baselines3
import sb3_contrib

from utils import FEATURE_NAMES
#from spca_selector_backup import SimpleSPCASelector, load_rollout_buffer
from spca import spca_weights_improved as spca_weights_improved_fn
from ig_attribution import ig_value_attr


models = {
    "PPO": {"MlpPolicy"},
    "RecurrentPPO": {"MlpLstmPolicy"},
}

selection_methods = ["attention", "spca", "ig"]


class FeatureSelectionTrainer:
    """
    Trainer + helper do wyboru cech wykorzystujący:
      - logi Attention
      - SPCA / IG na rezerwuarze .npz
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        env: Optional[Any] = None,
        selection_method: str = "attention",
        n_features_to_keep: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = FEATURE_NAMES,
        base_path='.'
    ):
        self.random_state = 42
        self.model = None  # will be created in train_model
        self.args = args
        self.env = env
        self.selection_method = selection_method
        self.algo_name = args.algo if args is not None else None
        self.AlgoClass = None
        self.n_features_to_keep = n_features_to_keep
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.tensorboard_log = "./output_malota/"
        self.base_path = base_path
        self.scores = None

    # ------------------------------------------------------------------
    # Trening
    # ------------------------------------------------------------------

    def train_model(self):
        if self.algo_name is None:
            raise ValueError("args.algo must be set before training.")
        try:
            self.AlgoClass = getattr(stable_baselines3, self.algo_name)
        except AttributeError:
            self.AlgoClass = getattr(sb3_contrib, self.algo_name)

        # Budowa modelu
        if self.selection_method == "attention":
            self.build_attention_model()
        else:
            self.build_ppo_model()

        # Tworzenie callbacków
        from callbacks import (
            ReservoirFromRolloutCallback,
            AttentionFromRolloutCallback,
            RollingRolloutBufferCallback
        )

        callbacks: List[Any] = []

        if self.selection_method == "attention":
            callbacks.append(AttentionFromRolloutCallback(
    compute_every_rollouts=1,      # co ile rolloutów liczyć statystyki
    warmup_rollouts=2,             # ile rolloutów pominąć na rozgrzewkę
    reservoir_size=500,            # ile PEŁNYCH epizodów trzyma rolling buffer
    select_k=7,                 # jak ustawisz np. 64 -> próba maskowania do 64 cech
    apply_mask=False,              # True jeśli chcesz realnie przycinać cechy
    feature_names=FEATURE_NAMES,            # albo lista nazw cech długości N
    print_every_steps=50_000,     # jak często wypisywać ranking
    save_npz_path=f"{self.base_path}/logs/spca_corr_attn_all_{self.args.model_name}/attn_cumulative_final.npz",
    rank_source="contrib",         # "contrib" lub "metric"
    mask_source="contrib",         # od czego zależy maska
    verbose=1,
))

        else:
            cb_spca = None

            # if "spca" in self.selection_method:
            #     cb_spca = ReservoirFromRolloutCallback(
            #         compute_every_rollouts=1,
            #         warmup_rollouts=2,
            #         print_every_steps=50_000,
            #         print_top_k=7,
            #         reservoir_size=70_000,
            #         n_components=4,
            #         alpha=1.0,
            #         ridge_alpha=0.01,
            #         max_iter=2000,
            #         tol=1e-6,
            #         method="cd",
            #         weight_norm="l1",
            #         normalize_weights=True,
            #         feature_names=[f"f{i}" for i in range(self.env.observation_space.shape[-1])],
            #         save_dir=f"{self.base_path}/logs/spca_corr_attn_all_{self.args.model_name}/",
            #         tensorboard=True,
            #         verbose=1,
            #     )
            cb_rolling = RollingRolloutBufferCallback(
    buffer_size=70_000,                      # keep last 50 episodes
    save_path=f"rollouts_buffer_{self.args.model_name}.pkl",     # where to save at the end
    save_on_training_end=True,
    verbose=1,
)
            callbacks = [cb for cb in (cb_spca, cb_rolling) if cb is not None]

        # Train
        self.model.learn(total_timesteps=self.args.num_timesteps, callback=callbacks)


    def get_model(self):
        return self.model
    # ------------------------------------------------------------------
    # Publiczny punkt wejścia do selekcji cech
    # ------------------------------------------------------------------

    def select_features(self):
        """
        Uruchom wybór cech po treningu.
        - Jeśli selection_method == "attention", czyta plik .npz z logami Attention
        - W przeciwnym razie używa bufora + SPCA / IG
        """
        selected_indices = None
        selected_names = None

        if self.selection_method == "attention":
            result = self.select_from_attn_npz(
                npz_path=f"{self.base_path}/logs/spca_corr_attn_all_{self.args.model_name}/attn_cumulative_final.npz",
                source="contrib",
                mode="cumulative",
                tau=0.9,
                k=7,
                bar_value="mean",
                plot=True,
                save_path=f"{self.base_path}/logs/topk_contrib_mean.png",
                show=False,
            )
            selected_indices = result["indices"]
            selected_names = result["feature_names"]
        else:
            # Użyj selekcji opartej na buforze zamiast rezerwuaru
            buffer_path = f"rollouts_buffer_{self.args.model_name}.pkl"
            result = self.select_from_buffer(
                buffer_path=buffer_path,
                method=self.selection_method,
                mode="cumulative",
                tau=0.9,
                k=7,
                plot=True,
                top_k_plot=7,
            )
            selected_indices = result["indices"]
            selected_names = result["feature_names"]

        print(
            f"Selected indices: {selected_indices}, "
            f"selected feature names: {selected_names}, "
            f"based on: {self.selection_method}"
        )

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def build_ppo_model(self) -> None:
        """
        Zbuduj standardowy model PPO/RecurrentPPO.
        """
        policy = self.args.policy
        import torch.nn as nn

        if policy == "MlpPolicy":
            policy_kwargs = dict(
                net_arch=[dict(pi=[32, 32], vf=[32, 32])],
                activation_fn=nn.ReLU,
                ortho_init=True,
            )
            n_steps = 2048
            batch_size = 2048

        elif policy == "MlpLstmPolicy":
            policy_kwargs = dict(
                lstm_hidden_size=64,
                net_arch=[dict(pi=[16, 16], vf=[16, 16])],
                activation_fn=nn.ReLU,
                ortho_init=True,
            )
            n_steps = 256 * 4
            batch_size = 256
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        self.model = self.AlgoClass(
            policy=policy,
            env=self.env,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=0.00003,
            vf_coef=1,
            clip_range_vf=10.0,
            max_grad_norm=1,
            gamma=0.95,
            ent_coef=0.001,
            clip_range=0.05,
            verbose=1,
            seed=int(self.args.seed),
            tensorboard_log=self.tensorboard_log,
        )

    def build_attention_model(self):
        from attention import train_model

        self.model = train_model(self.env, self.args)

    # ------------------------------------------------------------------
    # Narzędzia dla skumulowanych logów Attention (.npz)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_attn_cumulative(npz_path: str, source: str = "contrib"):
        """
        Wczytaj skumulowane statystyki Attention z logs/attn_cumulative_final.npz.

        source: "contrib" (domyślnie) lub "metric"
        Zwraca: steps, final_mean, final_var, final_freq, feature_names
        """
        d = np.load(npz_path, allow_pickle=False)
        steps = d["steps"]
        names = d["feature_names"].astype(str)

        source = str(source).lower()
        if source not in {"contrib", "metric"}:
            raise ValueError("source must be 'contrib' or 'metric'.")

        means_key = f"means_{source}"
        vars_key = f"vars_{source}"
        freqs_key = f"freqs_{source}"

        if means_key in d.files and vars_key in d.files and freqs_key in d.files:
            means = d[means_key]  # [S, N]
            vars_ = d[vars_key]   # [S, N]
            freqs = d[freqs_key]  # [S, N]
        else:
            if all(k in d.files for k in ["means", "vars", "freqs"]):
                means, vars_, freqs = d["means"], d["vars"], d["freqs"]
            else:
                raise KeyError(
                    f"Expected '{means_key}/{vars_key}/{freqs_key}' "
                    f"or legacy 'means/vars/freqs'. Found: {list(d.files)}"
                )

        final_mean = means[-1]  # [N]
        final_var = vars_[-1]   # [N]
        final_freq = freqs[-1]  # [N]
        return steps, final_mean, final_var, final_freq, names

    @staticmethod
    def _barplot_top_k(
        values,
        names,
        order,
        k,
        ylabel,
        title,
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """
        Prosty wykres słupkowy top-k cech według globalnego porządku.
        """
        order = np.asarray(order)
        idx = order[:k]
        vals = values[idx]
        labels = [f"{names[i]}" for i in idx]

        plt.figure(figsize=(10, 3.5))
        plt.bar(range(k), vals)
        plt.xticks(range(k), labels, rotation=45, ha="right", fontsize=10)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

    # ------------------------------------------------------------------
    # Ogólne narzędzia do pracy z wynikami (scores)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_scores(scores, eps: float = 1e-12) -> np.ndarray:
        """
        Przytnij do nieujemnych, usuń NaN/Inf i znormalizuj L1 do sumy=1.
        Gdy łączna masa ≈0, zwraca wektor zer.
        """
        s = np.asarray(scores, dtype=float)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.maximum(s, 0.0)
        total = s.sum()
        if total <= eps:
            return np.zeros_like(s)
        return s / total

    @staticmethod
    def select_indices_by_cumulative(
        scores,
        tau: float = 0.95,
        normalize: bool = True,
        min_k: int = 3, 
        max_k: Optional[int] = None,
        return_mask: bool = False,
        return_details: bool = False,
    ):
        """
        Wybierz najmniejszy zbiór top cech, dla którego suma skumulowana >= tau.
        """
        s = np.asarray(scores, dtype=float)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s[s < 0] = 0.0

        if normalize:
            total = s.sum()
            s = s / total if total > 0 else np.zeros_like(s)

        N = s.size
        order = np.argsort(s)[::-1]
        s_sorted = s[order]
        cum = np.cumsum(s_sorted)

        if tau <= 0:
            k = min_k
        elif tau >= 1:
            k = N
        else:
            meets = cum >= tau
            k = (np.argmax(meets) + 1) if np.any(meets) else N

        if max_k is not None:
            k = min(k, max_k)
        k = max(min_k, k)

        selected_idx = order[:k]

        if return_mask or return_details:
            mask = np.zeros(N, dtype=bool)
            mask[selected_idx] = True

        if return_details:
            details = {
                "order": order,
                "s_norm": s,
                "cum_sorted": cum,
            }
            return (selected_idx, mask, details) if return_mask else (selected_idx, details)

        return (selected_idx, mask) if return_mask else selected_idx

    # ------------------------------------------------------------------
    # Wysokopoziomowe API: selekcja cech z logów Attention (.npz)
    # ------------------------------------------------------------------

    def select_from_attn_npz(
        self,
        npz_path: str,
        source: str = "contrib",
        mode: str = "topk",
        k: int = 7,
        tau: float = 0.95,
        min_k: int = 3,
        max_k: Optional[int] = None,
        plot: bool = True,
        bar_value: str = "mean",
        save_path: Optional[str] = None,
        show: bool = False,
        verbose: bool = True,
    ):
        """
        Użyj skumulowanych logów Attention (.npz), aby wybrać cechy.
        """
        steps, final_mean, final_var, final_freq, names = self._load_attn_cumulative(
            npz_path, source=source
        )

        if verbose:
            print(f"[{source}] sum(mean) = {float(final_mean.sum()):.6f}")
            print(f"Loaded {len(names)} features from {npz_path}")

        order = np.lexsort((final_var, -final_mean, -final_freq))
        N = len(order)

        bar_value = str(bar_value).lower()
        if bar_value == "freq":
            scores_raw = final_freq.astype(float)
        else:
            scores_raw = final_mean.astype(float)

        # Przekształć wyniki (opcjonalne spłaszczenie/wyostrzenie)
        scores = self._transform_scores(
            scores_raw,
            transform="log",
            alpha=1.0,
        )

        mode = str(mode).lower()
        if mode == "cumulative":
            # selected_idx, details = self.select_indices_by_cumulative(
            #     scores,
            #     tau=tau,
            #     normalize=True,
            #     min_k=min_k,
            #     max_k=max_k,
            #     return_details=True,
            # )
            selected_idx = self.select_indices_by_cumulative(self.scores)
            print(f'Selected indices: {selected_idx}')
            k_sel = selected_idx.size
        else:
            k = max(min_k, min(k, N))
            k_sel = k
            selected_idx = order[:k_sel]

        if verbose:
            print(f"Selected {k_sel} features (mode='{mode}'):")
            for r, i in enumerate(order[:k_sel], 1):
                flag = "*" if i in selected_idx else " "
                print(
                    f"{r:>2}{flag} {names[i]:<20} idx={i:<4}  "
                    f"freq={final_freq[i]:.3f}  mean={final_mean[i]:.6f}  var={final_var[i]:.6f}"
                )

        if plot:
            if bar_value == "freq":
                vals = final_freq
                ylabel = "Średnia ważność"
                title = f"Ważność cech wg Attention (freq)"
            else:
                vals = final_mean
                ylabel = "Średnia ważność"
                title = f"Średnia ważność cech wg wkładu cech i wag Attention"

            # narysuj wszystkie cechy w wybranej kolejności, nie tylko wybrane
            k_plot = len(names)

            self._barplot_top_k(
                values=vals,
                names=names,
                order=order,
                k=k_plot,
                ylabel=ylabel,
                title=title,
                save_path=save_path,
                show=show,
            )

        if self.feature_names is None:
            self.feature_names = list(names)

        selected_names = [names[i] for i in selected_idx]

        return {
            "indices": selected_idx,
            "scores": scores,
            "order": order,
            "feature_names": selected_names,
            "final_mean": final_mean,
            "final_var": final_var,
            "final_freq": final_freq,
        }



    # Pomocnicze funkcje IG przeniesione do ig_attribution.py

    # ------------------------------------------------------------------
    # Selekcja cech z bufora (SPCA / IG)
    # ------------------------------------------------------------------


    def select_from_buffer(
        self,
        buffer_path: str,
        method: Optional[str] = None,  # "spca" | "ig"
        mode: str = "cumulative",
        k: Optional[int] = None,
        tau: float = 0.90,
        min_k: int = 1,
        max_k: Optional[int] = None,
        spca_components: int = 4,
        spca_alpha: float = 0.1,
        ig_baseline: str = "mean",
        ig_steps: int = 50,
        ig_batch: int = 256,
        sample_frac: Optional[float] = None,
        n_last_rollouts: Optional[int] = None,
        seed: int = 0xFEEDFACE,
        verbose: bool = True,
        plot: bool = False,
        top_k_plot: int = 20,
        save_prefix: Optional[str] = None,
        compute_ig: bool = True,
    ) -> Dict[str, Any]:
        """
        Uruchom selekcję cech SPCA / IG używając bufora zapisanego przez RollingRolloutBufferCallback.

        Argumenty:
            buffer_path: Ścieżka do pliku .pkl z buforem rolloutów.
            method: Metoda selekcji cech: "spca" lub "ig". Gdy None, użyje self.selection_method.
            mode: Tryb wyboru cech:
                  - "cumulative": wybiera minimalne k spełniające próg pokrycia tau,
                  - "topk": wybiera dokładnie k cech.
            k: Liczba cech dla trybu "topk". Przy "cumulative" ignorowane poza ograniczeniami min_k/max_k.
            tau: Próg pokrycia skumulowanego (0–1) w trybie "cumulative".
            min_k: Minimalna liczba cech, które zawsze zostaną wybrane (dolne ograniczenie).
            max_k: Maksymalna liczba cech, które mogą zostać wybrane (górne ograniczenie); None = brak limitu.
            spca_components: Maksymalna liczba komponentów SPCA (n_components).
            spca_alpha: Współczynnik sparsowości SPCA (niższy => łagodniejsza sparsowość).
            ig_baseline: Rodzaj bazowej obserwacji dla IG: "mean" | "zeros" | "min" | "custom".
            ig_steps: Liczba kroków interpolacji (m_steps) dla IG.
            ig_batch: Rozmiar batcha obliczeń dla IG.
            sample_frac: Część próbek z bufora do losowego zsubsample’owania (0–1). None = użyj wszystkich.
            n_last_rollouts: Jeśli podano, użyj tylko ostatnich n rolloutów z bufora.
            seed: Ziarno RNG używane przy losowaniu (sample_frac).
            verbose: Czy wypisywać szczegółowe informacje i rankingi.
            plot: Czy rysować wykresy rankingów dla wybranej metody.
            top_k_plot: Ile czołowych cech pokazać na wykresie.
            save_prefix: Prefiks ścieżki zapisu wykresów (np. "logs/sel"). None = nie zapisuj.
            compute_ig: Czy obliczać IG (musi być True, jeśli method == "ig").

        Zwraca:
            Słownik z kluczami:
              - "indices": wybrane indeksy cech (np.ndarray[int]),
              - "scores": wektor użytych wyników (SPCA/IG),
              - "method": użyta metoda ("spca" lub "ig"),
              - "order": indeksy cech w kolejności malejącej wg scores,
              - "df": DataFrame z wynikami per cecha,
              - "meta": metadane o buforze (ścieżka, liczba rolloutów),
              - "feature_names": nazwy wybranych cech,
              - "components": macierz komponentów SPCA.
        """
        # Wczytaj obserwacje z bufora z opcjonalnym wyborem ostatnich N
        X, _, names_loaded, meta_buf = load_rollout_buffer(buffer_path, n_last=n_last_rollouts)
        if X.size == 0:
            raise RuntimeError(f"Buffer '{buffer_path}' is empty (after applying n_last_rollouts).")
        N = X.shape[1]
        names = np.array(self.feature_names, dtype=str) if (self.feature_names is not None and len(self.feature_names) == N) else names_loaded

        if sample_frac is not None and 0.0 < sample_frac < 1.0:
            rng = np.random.default_rng(seed)
            size = max(1, int(round(sample_frac * X.shape[0])))
            idx_rows = rng.choice(X.shape[0], size=size, replace=False)
            X_use = X[idx_rows]
            #X_use = X[-100000:]
            #idx_rows=[0]
        else:
            idx_rows = np.arange(X.shape[0])
            X_use = X

        # Oblicz SPCA używając implementacji ze spca.py z ustalonymi parametrami
        w_spca, comps = spca_weights_improved_fn(
            X_use,
            n_components=spca_components,
            alpha=spca_alpha,
            ridge_alpha=0.01,
            max_iter=2000,
            tol=1e-6,
            method="cd",
            min_nonzero_rate=0.0,
            min_prestd=0.0,
            clip_z=5.0,
            component_weight=None,
            activity_gamma=0.5,
            normalize_weights=True,
        )

        ig = None
        if compute_ig:
            if self.model is None:
                raise RuntimeError("Model is not set. Call train_model() first.")
            ig = ig_value_attr(
                self.model,
                X_use,
                baseline=ig_baseline,
                m_steps=ig_steps,
                batch_size=ig_batch,
            )

        df = pd.DataFrame(
            {
                "feature": names,
                "idx": np.arange(N),
                "spca_w": w_spca,
                "ig": ig if ig is not None else np.zeros_like(w_spca),
            }
        )
        df["spca_w_pct"] = df["spca_w"] / max(df["spca_w"].sum(), 1e-12)
        df["ig_pct"] = (
            df["ig"] / max(df["ig"].sum(), 1e-12) if compute_ig and ig is not None else np.zeros_like(df["ig"])
        )
        df_rank_spca = df.sort_values("spca_w", ascending=False).reset_index(drop=True)
        df_rank_ig = (
            df.sort_values("ig", ascending=False).reset_index(drop=True)
            if compute_ig and ig is not None
            else None
        )

        # Sterowanie printowaniem i rysowaniem wykresów
        m = (method or self.selection_method or "spca").lower()
        assert m in {"spca", "ig"}

        if verbose:
            k_show = min(top_k_plot, N)
            print(f"[Buffer] file={os.path.basename(buffer_path)}, rollouts={meta_buf.get('episodes','?')}, rows_used={len(idx_rows)}, N={N}")
            if m == "spca":
                print(f"Top-{k_show} cech wg wagi SPCA:")
                for i, row in df_rank_spca.head(k_show).iterrows():
                    print(
                        f"  {i+1:>2}. {row['feature']} (#{int(row['idx'])})  w={row['spca_w']:.6f}"
                    )
            elif m == "ig" and df_rank_ig is not None:
                print(f"\nTop-{k_show} cech wg IG (akcje):")
                for i, row in df_rank_ig.head(k_show).iterrows():
                    print(
                        f"  {i+1:>2}. {row['feature']} (#{int(row['idx'])})  ig={row['ig']:.6e}"
                    )

        if m == "spca":
            scores = df["spca_w"].to_numpy()
        else:
            if not compute_ig or ig is None:
                raise RuntimeError("IG scores are disabled or unavailable.")
            scores = df["ig"].to_numpy()

        scores = np.asarray(scores, dtype=float)
        self.scores = scores.copy()
        print(f'Scores: {scores}')
        order = np.argsort(-scores)


        # Wywołanie funkcji do wyboru top K
        if mode == "cumulative":
            selected_idx = self.select_indices_by_cumulative(self.scores)
            print(f'SELECTED IDX:,{selected_idx}')
            k_sel = selected_idx.size
        else:
            N_feat = scores.size
            k_sel = max(min_k, min((k or N_feat), N_feat))
            selected_idx = order[:k_sel]
        print(f'Selected indices: {selected_idx}')
        if verbose:
            print(
                f"\n[Selection/buffer] method={m}, mode={mode}, selected {k_sel} features"
            )
            for rank, i in enumerate(order[:k_sel], 1):
                star = "*" if i in selected_idx else " "
                print(
                    f"{rank:>2}{star} {names[i]:<20} idx={i:<4}  score={scores[i]:.6g}"
                )

        if self.feature_names is None:
            self.feature_names = list(names)
        selected_names = [names[i] for i in selected_idx]

        if plot:
            self._plot_buffer_rankings(
                df_rank_spca=(df_rank_spca if m == "spca" else None),
                df_rank_ig=(df_rank_ig if (m == "ig" and compute_ig and df_rank_ig is not None) else None),
                top_k=min(top_k_plot, N),
                save_prefix=save_prefix,
            )

        return {
            "indices": selected_idx,
            "scores": scores,
            "method": m,
            "order": order,
            "df": df,
            "meta": {"buffer_path": buffer_path, "rollouts": meta_buf.get('episodes', None)},
            "feature_names": selected_names,
            "components": comps,
        }

    @staticmethod
    def _plot_buffer_rankings(
        df_rank_spca: Optional[pd.DataFrame],
        df_rank_ig: Optional[pd.DataFrame],
        top_k: int = 20,
        save_prefix: Optional[str] = None,
    ) -> None:
        """
        Wykres top-k cech według wagi SPCA oraz IG dla danych z bufora.
        """

        def _barplot(series, labels, ylabel, title, fname=None, alpha=0.7, color="skyblue"):
            plt.figure(figsize=(10, 3.5))
            vals = series.values
            names_plot = [str(labels.iloc[i]) for i in range(len(series))]
            plt.bar(range(len(vals)), vals, color=color, alpha=alpha)
            plt.xticks(range(len(vals)), names_plot, rotation=45, ha="right", fontsize=10)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.tight_layout()
            if fname:
                plt.savefig(fname, dpi=150)
            plt.show()

        # Rysuj SPCA, jeśli przekazano
        if df_rank_spca is not None and len(df_rank_spca) > 0:
            k = min(top_k, len(df_rank_spca))
            _barplot(
                df_rank_spca.head(k)["spca_w"],
                df_rank_spca.head(k)["feature"],
                "wagi SPCA",
                f"Średnia bezwzględna ważność cech wg SPCA",
                fname=f"{save_prefix}_spca.png" if save_prefix else None,
                color="black",
                alpha=0.5,
            )

        # Rysuj IG, jeśli przekazano
        if df_rank_ig is not None and len(df_rank_ig) > 0:
            k_ig = min(top_k, len(df_rank_ig))
            _barplot(
                df_rank_ig.head(k_ig)["ig"],
                df_rank_ig.head(k_ig)["feature"],
                "|IG| (akcje)",
                f"Średnia bezwzględna ważność cech wg IG (akcje)",
                fname=f"{save_prefix}_ig.png" if save_prefix else None,
                color="steelblue",
                alpha=1.0,
            )

    # ------------------------------------------------------------------
    # Pomocniczy kod przekształcania wyników (dla Attention)
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_scores(
        scores,
        transform: str = "none",  # "none" | "power" | "log" | "sqrt"
        alpha: float = 1.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Zastosuj monotoniczne przekształcenie wyników przed selekcją.
        """
        s = np.asarray(scores, dtype=float)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s[s < 0.0] = 0.0

        t = transform.lower()
        if t == "none":
            pass
        elif t == "power":
            s = np.power(s, alpha)
        elif t == "log":
            s = np.log1p(s)
        elif t == "sqrt":
            s = np.sqrt(s)
        else:
            raise ValueError(f"Unknown score transform: {transform}")

        return s


def load_rollout_buffer(path: str, n_last: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, int]]:
    """
    Wczytaj bufor rolloutów (.pkl) i zwróć sklejone obserwacje.

    Zwraca:
        X  : np.ndarray [R, N] – sklejone obserwacje z wszystkich epizodów
        y  : None (miejsce na ewentualne cele – tutaj pomijamy)
        names : np.ndarray [N] – nazwy cech (FEATURE_NAMES lub f{i})
        meta  : dict – {'episodes', 'rows', 'source_file'}
    """
    with open(path, "rb") as f:
        rollouts = pickle.load(f)
    if not isinstance(rollouts, list) or len(rollouts) == 0:
        return np.zeros((0, 0), dtype=np.float32), None, np.array([], dtype=str), {
            "episodes": 0,
            "rows": 0,
            "source_file": os.path.basename(path),
        }
    # Użyj tylko N ostatnich rolloutów, jeśli podano
    if n_last is not None:
        try:
            k = int(n_last)
        except Exception:
            k = None
        if k is not None:
            if k <= 0:
                rollouts = []
            elif k < len(rollouts):
                rollouts = rollouts[-k:]
    obs_rows: List[np.ndarray] = []
    for ep in rollouts:
        obs = np.asarray(ep.get("obs", np.zeros((0,))), dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_rows.append(obs)
    X = np.concatenate(obs_rows, axis=0) if obs_rows else np.zeros((0, 0), dtype=np.float32)
    N = X.shape[1] if X.ndim == 2 and X.size > 0 else 0
    # Informacyjny wydruk rozmiaru po ewentualnym przycięciu i sklejaniu
    print(f"[load_rollout_buffer] episodes={len(rollouts)}, samples={X.shape[0]}, N={N}")
    print(f"[load_rollout_buffer] episodes={len(rollouts)}, samples={X.shape[0]}, N={X.shape[1] if X.ndim==2 else 0}")
    if N > 0 and isinstance(FEATURE_NAMES, (list, tuple)) and len(FEATURE_NAMES) == N:
        names = np.asarray(FEATURE_NAMES, dtype=str)
    else:
        names = np.array([f"f{i}" for i in range(N)], dtype=str)
    meta = {
        "episodes": len(rollouts),
        "rows": int(X.shape[0]),
        "source_file": os.path.basename(path),
    }
    return X, None, names, meta
