import numpy as np
import torch as th
from typing import Any, Dict, Optional, Sequence, Tuple

from sklearn.decomposition import SparsePCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import stable_baselines3
import sb3_contrib
import os
import matplotlib.pyplot as plt
import pandas as pd

from utils import FEATURE_NAMES


models = {
    'PPO': {
        'MlpPolicy'
    },
    'RecurrentPPO': {
        'MlpLstmPolicy'
    },
}

selection_methods = ["attention", "correlation", "spca", "ig"]


import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import pandas as pd

import stable_baselines3
import sb3_contrib

from utils import FEATURE_NAMES


# Optional: just for reference / validation elsewhere
selection_methods = ["attention", "correlation", "spca", "ig"]


class FeatureSelectionTrainer:
    """
    Trainer + feature selection helper.

    - Trains an SB3/sb3-contrib model (PPO / RecurrentPPO) with optional attention/SPCA/corr callbacks.
    - Supports post-hoc feature selection via:
        * attention cumulative logs (.npz)
        * SPCA / correlation / IG on a reservoir (.npz)
    """

    def __init__(
        self,
        args=None,
        env=None,
        selection_method: str = "attention",
        n_features_to_keep: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.random_state = 42
        self.model = None  # will be created in train_model
        self.args = args
        self.env = env
        self.selection_method = selection_method
        self.algo_name = args.algo if args is not None else None
        self.AlgoClass = None
        self.n_features_to_keep = n_features_to_keep
        self.feature_names = list(feature_names) if feature_names is not None else FEATURE_NAMES
        self.tensorboard_log = "./output_malota/"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(self):
        if self.algo_name is None:
            raise ValueError("args.algo must be set before training.")

        # Resolve algo class (SB3 or sb3_contrib)
        try:
            self.AlgoClass = getattr(stable_baselines3, self.algo_name)
        except AttributeError:
            self.AlgoClass = getattr(sb3_contrib, self.algo_name)

        # Build model
        if self.selection_method == "attention":
            self.build_attention_model()
        else:
            self.build_ppo_model()

        # Build callbacks
        from callbacks import (
            FeatureCorrelationFromRolloutCallback,
            FeatureSPCAFromRolloutCallback,
            AttentionFromRolloutCallback,
        )

        callbacks = []

        if self.selection_method == "attention":
            callbacks.append(
                AttentionFromRolloutCallback(
                    compute_every_rollouts=1,
                    warmup_rollouts=2,
                    print_every_steps=50_000,
                    print_top_k=7,
                    top_m_for_frequency=None,
                    feature_names=FEATURE_NAMES,
                    select_k=None,
                    corr_threshold=0.95,
                    reservoir_size=20_000,
                    rank_source="contrib",
                    mask_source="contrib",
                    save_npz_path=(
                        f"logs/spca_corr_attn_all_{self.args.model_name}/"
                        f"attn_cumulative_final.npz"
                    ),
                    verbose=1,
                )
            )
        else:
            cb_spca = None
            cb_corr = None

            if "spca" in self.selection_method:
                cb_spca = FeatureSPCAFromRolloutCallback(
                    compute_every_rollouts=1,
                    warmup_rollouts=2,
                    print_every_steps=50_000,
                    print_top_k=7,
                    reservoir_size=20_000,
                    n_components=3,
                    alpha=1.0,
                    ridge_alpha=0.01,
                    max_iter=1000,
                    tol=1e-8,
                    method="lars",
                    weight_norm="l1",
                    normalize_weights=True,
                    feature_names=[
                        f"f{i}" for i in range(self.env.observation_space.shape[-1])
                    ],
                    save_dir=f"./logs/spca_corr_attn_all_{self.args.model_name}/",
                    tensorboard=True,
                    verbose=1,
                    final_basename="spca_final",
                )

            if "correlation" in self.selection_method:
                cb_corr = FeatureCorrelationFromRolloutCallback(
                    compute_every_rollouts=1,
                    warmup_rollouts=2,
                    print_every_steps=50_000,
                    print_top_k=7,
                    redundancy_threshold=0.95,
                    reservoir_size=20_000,
                    target_kind="advantage",  # or "return" etc.
                    feature_names=[
                        f"f{i}" for i in range(self.env.observation_space.shape[-1])
                    ],
                    save_dir=f"./logs/spca_corr_attn_all_{self.args.model_name}/",
                    tensorboard=True,
                    verbose=1,
                )

            callbacks = [cb for cb in (cb_spca, cb_corr) if cb is not None]

        # Train
        self.model.learn(total_timesteps=self.args.total_timesteps, callback=callbacks)

    # ------------------------------------------------------------------
    # Public selection entrypoint
    # ------------------------------------------------------------------

    def select_features(self):
        """
        Run feature selection after training.

        - If selection_method == 'attention': uses attention cumulative .npz
        - Else (spca / correlation / ig): uses reservoir .npz and
          SPCA / corr / IG functions from compare_spca_corr_ig.py
        """
        selected_indices = None
        selected_names = None

        if self.selection_method == "attention":
            result = self.select_from_attn_npz(
                npz_path=(
                    f"logs/spca_corr_attn_all_{self.args.model_name}/"
                    f"attn_cumulative_final.npz"
                ),
                source="contrib",      # or "metric"
                mode="cumulative",
                tau=0.95,
                k=7,
                bar_value="mean",
                plot=True,
                save_path=(
                    f"logs/spca_corr_attn_all_{self.args.model_name}/"
                    f"topk_contrib_mean.png"
                ),
                show=False,
            )
            selected_indices = result["indices"]
            selected_names = result["feature_names"]

        else:
            reservoir_path = (
                f"./logs/spca_corr_attn_all_{self.args.model_name}/"
                f"fcorr_reservoir_final.npz"
            )

            result = self.select_from_reservoir(
                reservoir_path=reservoir_path,
                method=self.selection_method,  # "spca", "correlation", or "ig"
                mode="cumulative",
                tau=0.95,
                k=7,
                min_k=3,
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
        Build a standard PPO/RecurrentPPO model.
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
            vf_coef=1.0,
            clip_range_vf=10.0,
            max_grad_norm=1.0,
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
    # Attention cumulative .npz utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _load_attn_cumulative(npz_path: str, source: str = "contrib"):
        """
        Load cumulative attention stats from logs/attn_cumulative_final.npz.

        source: "contrib" (default) or "metric"
        Returns: steps, final_mean, final_var, final_freq, feature_names
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
        Simple bar plot of top-k features given a global ordering.
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
    # Generic score utilities
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_scores(scores, eps: float = 1e-12) -> np.ndarray:
        """
        Clamp to non-negative, sanitize NaNs/Infs, and L1-normalize to sum=1.
        If total mass is ~0, returns an all-zeros vector.
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
        min_k: int = 3,  # default: at least 3 features
        max_k: Optional[int] = None,
        return_mask: bool = False,
        return_details: bool = False,
    ):
        """
        Select the smallest set of top features whose cumulative score >= tau.

        Args:
            scores: array-like, per-feature scores (can be unnormalized).
            tau: coverage threshold in [0,1].
            normalize: if True, L1-normalize scores before coverage.
            min_k: lower bound on number of features to select.
            max_k: optional cap on number of features to select.
            return_mask: if True, also return a boolean mask.
            return_details: if True, also return dict with order/s_norm/cum.

        Returns:
            selected_idx  (np.ndarray, shape [k], dtype=int)
            [mask]        (optional, boolean array shape [N])
            [details]     (optional, dict with 'order', 's_norm', 'cum_sorted')
        """
        s = np.asarray(scores, dtype=float)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s[s < 0] = 0.0

        if normalize:
            total = s.sum()
            s = s / total if total > 0 else np.zeros_like(s)

        N = s.size
        order = np.argsort(s)[::-1]       # descending
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

        mask = None
        if return_mask or return_details:
            mask = np.zeros(N, dtype=bool)
            mask[selected_idx] = True

        if return_details:
            details = {
                "order": order,
                "s_norm": s,
                "cum_sorted": cum,
            }
            if return_mask:
                return selected_idx, mask, details
            return selected_idx, details

        if return_mask:
            return selected_idx, mask
        return selected_idx

    @staticmethod
    def _transform_scores(
        scores,
        transform: str = "none",   # "none" | "power" | "log" | "sqrt"
        alpha: float = 1.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Apply a monotonic transform to scores before selection.

        - "none":  no change
        - "power": s -> s^alpha   (alpha<1 flattens, alpha>1 sharpens)
        - "log":   s -> log1p(s)
        - "sqrt":  s -> sqrt(s)

        Returned scores are non-negative; *not* normalized here.
        """
        s = np.asarray(scores, dtype=float)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        s[s < 0.0] = 0.0

        t = str(transform).lower()
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

    # ------------------------------------------------------------------
    # High-level API: feature selection from attention .npz logs
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
        Use cumulative attention logs (.npz) to select features.
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

        # Optionally transform attention scores before selection
        scores = self._transform_scores(
            scores_raw,
            transform="log",  # or "none" / "sqrt" / "power"
            alpha=1.0,
        )

        mode = str(mode).lower()
        if mode == "cumulative":
            selected_idx, details = self.select_indices_by_cumulative(
                scores,
                tau=tau,
                normalize=True,
                min_k=min_k,
                max_k=max_k,
                return_details=True,
            )
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
        plot=True
        if plot:
            if bar_value == "freq":
                vals = final_freq
                ylabel = "Średnia ważność"
                title = "Ważność cech wg Attention (freq)"
            else:
                vals = final_mean
                ylabel = "Średnia ważność"
                title = "Ważność cech wg Attention (mean)"

            self._barplot_top_k(
                values=vals,
                names=names,
                order=order,
                k=k_sel,
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

    # ------------------------------------------------------------------
    # High-level: feature selection from reservoir (SPCA / corr / IG)
    # ------------------------------------------------------------------

    def select_from_reservoir(
        self,
        reservoir_path: str,
        method: Optional[str] = None,   # "spca" | "correlation" | "ig"
        mode: str = "topk",             # "topk" | "cumulative"
        k: int = 20,
        tau: float = 0.90,
        min_k: int = 1,
        max_k: Optional[int] = None,
        spca_components: int = 3,
        spca_alpha: float = 1.0,
        corr_name: str = "advantage",
        ig_baseline: str = "mean",
        ig_steps: int = 50,
        ig_batch: int = 256,
        sample_size: Optional[int] = None,
        seed: int = 0xC0FFEE,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run SPCA / correlation / IG feature selection using a reservoir .npz file.

        Uses load_reservoir, spca_weights_backup, corr_feature_target,
        and ig_value_attr from compare_spca_corr_ig.py for ranking.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained (self.model is None).")

        # 1) Load reservoir
        X, y, names_saved, meta = load_reservoir(reservoir_path)
        R, N = X.shape

        if self.feature_names is not None and len(self.feature_names) == N:
            names = np.array(self.feature_names, dtype=str)
        else:
            names = names_saved

        # 2) Subsample rows consistently across methods (optional)
        if sample_size is not None and sample_size < R:
            rng = np.random.default_rng(seed)
            idx_rows = rng.choice(R, size=sample_size, replace=False)
            X_use = X[idx_rows]
            y_use = y[idx_rows] if y is not None else None
        else:
            idx_rows = np.arange(R)
            X_use, y_use = X, y

        # 3) SPCA weights via spca_weights_backup
        w_spca, comps = spca_weights_backup(
            X_use,
            n_components=spca_components,
            alpha=spca_alpha,
        )

        # 4) Correlations (if y available)
        r = corr_feature_target(X_use, y_use) if y_use is not None else None

        # 5) IG (action-based) via ig_value_attr
        ig = ig_value_attr(
            self.model,
            X_use,
            baseline=ig_baseline,
            m_steps=ig_steps,
            batch_size=ig_batch,
            progress=verbose,
        )

        # 6) Build comparison table (same spirit as compare_spca_corr_ig)
        df = pd.DataFrame(
            {
                "feature": names,
                "idx": np.arange(N),
                "spca_w": w_spca,
                "ig": ig,
            }
        )
        if r is not None:
            df["corr"] = r
            df["abs_corr"] = np.abs(r)

        df["spca_w_pct"] = df["spca_w"] / max(df["spca_w"].sum(), 1e-12)
        df["ig_pct"] = df["ig"] / max(df["ig"].sum(), 1e-12)
        if r is not None:
            df["abs_corr_pct"] = df["abs_corr"] / max(df["abs_corr"].sum(), 1e-12)

        df_rank_spca = df.sort_values("spca_w", ascending=False).reset_index(drop=True)
        df_rank_ig = df.sort_values("ig", ascending=False).reset_index(drop=True)
        df_rank_corr = (
            df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
            if r is not None
            else None
        )

        if verbose:
            k_show = min(k, N)
            print(
                f"[Reservoir] file={meta['source_file']}, "
                f"rows_used={len(idx_rows)}, N={N}"
            )
            print(f"Top-{k_show} by SPCA weight:")
            for i, row in df_rank_spca.head(k_show).iterrows():
                print(
                    f"  {i+1:>2}. {row['feature']} (#{int(row['idx'])})  "
                    f"w={row['spca_w']:.6f}"
                )
            print(f"\nTop-{k_show} by IG (value):")
            for i, row in df_rank_ig.head(k_show).iterrows():
                print(
                    f"  {i+1:>2}. {row['feature']} (#{int(row['idx'])})  "
                    f"ig={row['ig']:.6e}"
                )
            if df_rank_corr is not None:
                print(f"\nTop-{k_show} by |corr(feature, {corr_name})|:")
                for i, row in df_rank_corr.head(k_show).iterrows():
                    print(
                        f"  {i+1:>2}. {row['feature']} (#{int(row['idx'])})  "
                        f"r={float(row['corr']):+.4f}  |r|={row['abs_corr']:.4f}"
                    )

        # 7) Choose which metric drives selection
        m = (method or self.selection_method or "spca").lower()
        assert m in {"spca", "correlation", "ig"}

        if m == "spca":
            scores = df["spca_w"].to_numpy()
        elif m == "ig":
            scores = df["ig"].to_numpy()
        else:  # "correlation"
            if r is None:
                raise RuntimeError(
                    "Reservoir does not have y; cannot compute correlation-based scores."
                )
            scores = df["abs_corr"].to_numpy()

        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores)  # best first

        # 8) Decide which indices to keep
        mode = str(mode).lower()
        if mode == "cumulative":
            selected_idx, details = self.select_indices_by_cumulative(
                scores,
                tau=tau,
                normalize=True,
                min_k=min_k,
                max_k=max_k,
                return_details=True,
            )
            k_sel = selected_idx.size
        else:
            N_feat = scores.size
            k_sel = max(min_k, min(k, N_feat))
            selected_idx = order[:k_sel]

        if verbose:
            print(
                f"\n[Selection] method={m}, mode={mode}, selected {k_sel} "
                f"features"
            )
            for rank, i in enumerate(order[:k_sel], 1):
                star = "*" if i in selected_idx else " "
                print(
                    f"{rank:>2}{star} {names[i]:<20} idx={i:<4}  "
                    f"score={scores[i]:.6g}"
                )

        if self.feature_names is None:
            self.feature_names = list(names)
        selected_names = [names[i] for i in selected_idx]
        plot=True
        if plot:
            k_plot = N
            _plot_reservoir_rankings(
                df_rank_spca=df_rank_spca,
                df_rank_ig=df_rank_ig,
                df_rank_corr=df_rank_corr,
                k_plot=k_plot,
                save_prefix=f'{self.args.model_name}',
            )

        return {
            "indices": selected_idx,
            "scores": scores,
            "method": m,
            "order": order,
            "df": df,
            "meta": meta,
            "feature_names": selected_names,
        }



# compare_spca_corr_ig.py
import os
from typing import Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from sklearn.decomposition import SparsePCA

# ---------------- utils: policy type ----------------

def _is_recurrent(policy) -> bool:
    return hasattr(policy, "lstm_actor") and hasattr(policy, "lstm_critic")

@th.no_grad()
def _make_lstm_states(policy, batch: int, device: th.device):
    """
    Build LSTM states in the structure expected by sb3-contrib:
      RecurrentStates(pi=(h_a,c_a), vf=(h_c,c_c))
    Falls back to a SimpleNamespace with .pi/.vf if import path changes.
    """
    la, lc = policy.lstm_actor, policy.lstm_critic
    h_a = th.zeros(la.num_layers, batch, la.hidden_size, device=device)
    c_a = th.zeros_like(h_a)
    h_c = th.zeros(lc.num_layers, batch, lc.hidden_size, device=device)
    c_c = th.zeros_like(h_c)
    try:
        # SB3-contrib >= 2.x
        from sb3_contrib.common.recurrent.type_aliases import RecurrentStates
        return RecurrentStates(pi=(h_a, c_a), vf=(h_c, c_c))
    except Exception:
        # Fallback: struct with .pi and .vf attrs
        from types import SimpleNamespace
        return SimpleNamespace(pi=(h_a, c_a), vf=(h_c, c_c))

# ---------------- reservoir loader ----------------

def load_reservoir(path: str):
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)                    # [R, N]
    y = d["y"].astype(np.float32) if "y" in d.files else None  # [R] or None
    names = d["feature_names"].astype(str) if "feature_names" in d.files else np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)
    meta = {
        "rows_filled": int(d["rows_filled"]) if "rows_filled" in d.files else None,
        "rows_seen":   int(d["rows_seen"])   if "rows_seen"   in d.files else None,
        "step":        int(d["step"])        if "step"        in d.files else None,
        "source_file": os.path.basename(path),
    }
    return X, y, names, meta

# --------------- SPCA weights (top-K) ---------------

def _zscore_cols(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0, keepdims=True)
    Z = X - mu
    sd = np.nanstd(Z, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return Z / sd

def spca_weights_backup(X: np.ndarray, n_components: int = 3, alpha: float = 1.0, ridge_alpha: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-8, method: str = "lars",
                 weight_norm: str = "l1", normalize_weights: bool = True, random_state: int = 0xC0FFEE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (weights [N], components [K,N])
    """
    Z = _zscore_cols(X)
    K = min(n_components, Z.shape[1])
    spca = SparsePCA(n_components=K, alpha=alpha, ridge_alpha=ridge_alpha, max_iter=max_iter,
                     tol=tol, method=method, random_state=random_state)
    spca.fit(Z)
    comps = spca.components_.astype(np.float64, copy=False)   # [K, N]
    # L2-normalize each component row for stable aggregation
    norms = np.linalg.norm(comps, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    comps = comps / norms
    if weight_norm == "l1":
        w = np.sum(np.abs(comps), axis=0)
    else:
        w = np.sqrt(np.sum(comps ** 2), axis=0)  # NOTE: equivalent to L2 over components
    # Fix: previous line should be along axis=0
    w = np.sqrt(np.sum(comps ** 2, axis=0)) if weight_norm != "l1" else w

    w = w.astype(np.float64)
    if normalize_weights and w.sum() > 0:
        w = w / w.sum()
    return w.astype(np.float32), comps.astype(np.float32)

# --------------- Correlation r(feature, y) ---------------

def corr_feature_target(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False).reshape(-1)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    y = y - np.nanmean(y)
    sx = np.nanstd(X, axis=0)
    sy = float(np.nanstd(y))
    sx[sx == 0] = np.nan
    sy = sy if sy != 0 else np.nan
    r = (X * y[:, None]).sum(axis=0) / ((X.shape[0] - 1) * sx * sy)
    r = np.clip(np.nan_to_num(r, nan=0.0), -1.0, 1.0)
    return r.astype(np.float32)

# --------------- IG for critic value V(s) ---------------

# def ig_value_attr(model, X: np.ndarray, baseline: str = "mean", baseline_vec: Optional[np.ndarray] = None,
#                   m_steps: int = 50, batch_size: int = 256, device: Optional[str] = None, progress: bool = True) -> np.ndarray:
#     """
#     Returns mean|IG| per feature over rows in X: [N]
#     Works for PPO(MlpPolicy) and RecurrentPPO(MlpLstmPolicy).
#     """
#     policy = model.policy
#     policy.eval()
#     dev = th.device(device or policy.device)
#     Xf = th.tensor(X, dtype=th.float32, device=dev)
#     R, N = Xf.shape

#     # baseline
#     if baseline == "zeros":
#         b = th.zeros(N, device=dev)
#     elif baseline == "mean":
#         b = Xf.mean(dim=0)
#     elif baseline == "min":
#         b = Xf.min(dim=0).values
#     elif baseline == "custom":
#         assert baseline_vec is not None and baseline_vec.shape == (N,)
#         b = th.tensor(baseline_vec, dtype=th.float32, device=dev)
#     else:
#         raise ValueError("baseline must be 'zeros'|'mean'|'min'|'custom'")

#     alphas = th.linspace(1.0 / m_steps, 1.0, m_steps, device=dev)
#     IG_abs_sum = th.zeros(N, dtype=th.float32, device=dev)
#     total_rows = 0
#     recurrent = _is_recurrent(policy)

#     if progress:
#         print(f"[IG] rows={R}, m_steps={m_steps}, batch={batch_size}, baseline={baseline}, recurrent={recurrent}")

#     for start in range(0, R, batch_size):
#         end = min(start + batch_size, R)
#         xb = Xf[start:end]  # [B, N]
#         B = b.unsqueeze(0).expand_as(xb)
#         total_grad = th.zeros_like(xb)

#         for a in alphas:
#             x_alpha = (B + a * (xb - B)).clone().detach().requires_grad_(True)

#             if recurrent:
#                 # IMPORTANT: float32 episode_starts to avoid "1.0 - bool" runtime error in older sb3-contrib
#                 ep_starts = th.ones(x_alpha.shape[0], dtype=th.float32, device=dev)
#                 lstm_states = _make_lstm_states(policy, x_alpha.shape[0], dev)
#                 out = policy.forward(x_alpha, lstm_states, ep_starts, deterministic=True)
#                 values = out[1]  # [B, 1]
#             else:
#                 # MLP policy: forward(obs, deterministic) -> (actions, values, log_prob)
#                 out = policy.forward(x_alpha, deterministic=True)
#                 values = out[1]  # [B, 1]

#             v_sum = values.sum()
#             grads = th.autograd.grad(v_sum, x_alpha, retain_graph=False, create_graph=False)[0]
#             total_grad += grads

#         ig_batch = (xb - B) * (total_grad / m_steps)  # [B, N]
#         IG_abs_sum += ig_batch.abs().sum(dim=0)
#         total_rows += (end - start)

#         if progress:
#             print(f"  processed {end}/{R}", end="\r")
#     if progress:
#         print()

#     mean_abs_ig = (IG_abs_sum / max(1, total_rows)).detach().cpu().numpy()
#     return mean_abs_ig.astype(np.float32)

def ig_value_attr(
    model,
    X: np.ndarray,
    baseline: str = "mean",
    baseline_vec: Optional[np.ndarray] = None,
    m_steps: int = 50,
    batch_size: int = 256,
    device: Optional[str] = None,
    progress: bool = True,
) -> np.ndarray:
    """
    ACTION-focused Integrated Gradients (IG) for the policy:
      objective at each interpolation point x_alpha is sum(log π(a* | x_alpha)),
      where a* is the policy's deterministic action at the endpoint x.

    Returns mean|IG| per feature over rows in X: [N]
    Works for PPO(MlpPolicy) and RecurrentPPO(MlpLstmPolicy).
    Requires helpers: `_is_recurrent(policy)` and `_make_lstm_states(policy, batch, device)`.
    """
    policy = model.policy
    policy.eval()
    dev = th.device(device or policy.device)
    Xf = th.tensor(X, dtype=th.float32, device=dev)
    R, N = Xf.shape

    # ---- baseline vector ----
    if baseline == "zeros":
        b = th.zeros(N, device=dev)
    elif baseline == "mean":
        b = Xf.mean(dim=0)
    elif baseline == "min":
        b = Xf.min(dim=0).values
    elif baseline == "custom":
        assert baseline_vec is not None and baseline_vec.shape == (N,)
        b = th.tensor(baseline_vec, dtype=th.float32, device=dev)
    else:
        raise ValueError("baseline must be 'zeros'|'mean'|'min'|'custom'")

    alphas = th.linspace(1.0 / m_steps, 1.0, m_steps, device=dev)
    IG_abs_sum = th.zeros(N, dtype=th.float32, device=dev)
    total_rows = 0
    recurrent = _is_recurrent(policy)

    if progress:
        print(f"[IG(action)] rows={R}, m_steps={m_steps}, batch={batch_size}, "
              f"baseline={baseline}, recurrent={recurrent}")

    # ---- helper to get action distribution ----
    def _get_action_dist(obs_t: th.Tensor, lstm_states=None, ep_starts=None):
        """
        Return a distribution object for actions given obs_t.

        - For MlpPolicy: use policy.get_distribution(...) or manual MLP path.
        - For RecurrentPPO/MlpLstmPolicy: features -> LSTM -> mlp_extractor -> dist.
        """
        if not recurrent:
            # SB3 >= 1.8: MlpPolicy has get_distribution
            try:
                return policy.get_distribution(obs_t)
            except Exception:
                # Fallback: manual path (shared MLP)
                features = policy.extract_features(obs_t)
                latent_pi, _ = policy.mlp_extractor(features)
                return policy._get_action_dist_from_latent(latent_pi)

        # Recurrent policy (e.g. RecurrentPPO with MlpLstmPolicy)
        features = policy.extract_features(obs_t)  # [B, features_dim]

        if lstm_states is None:
            lstm_states = _make_lstm_states(policy, obs_t.shape[0], dev)
        if ep_starts is None:
            ep_starts = th.ones(obs_t.shape[0], dtype=th.float32, device=dev)

        # 1) through LSTM actor
        lstm_out_pi, _ = policy._process_sequence(
            features, lstm_states.pi, ep_starts, policy.lstm_actor
        )  # [B, lstm_hidden_size]

        # 2) through MLP extractor (pi branch)
        latent_pi, _ = policy.mlp_extractor(lstm_out_pi)  # [B, latent_dim_pi]

        # 3) distribution from latent
        dist = policy._get_action_dist_from_latent(latent_pi)
        return dist

    # ---- main IG loop ----
    for start in range(0, R, batch_size):
        end = min(start + batch_size, R)
        xb = Xf[start:end]  # [B, N]
        B = b.unsqueeze(0).expand_as(xb)

        # ---- build fixed a* (deterministic action at endpoint xb) ----
        with th.no_grad():
            if recurrent:
                ep_starts_b = th.ones(xb.shape[0], dtype=th.float32, device=dev)
                lstm_b = _make_lstm_states(policy, xb.shape[0], dev)
                dist_b = _get_action_dist(xb, lstm_states=lstm_b, ep_starts=ep_starts_b)
            else:
                dist_b = _get_action_dist(xb)
            a_star = dist_b.get_actions(deterministic=True)  # [B, action_dim]

        total_grad = th.zeros_like(xb)

        # ---- IG accumulation along the path ----
        for a in alphas:
            x_alpha = (B + a * (xb - B)).clone().detach().requires_grad_(True)
            if recurrent:
                ep_starts = th.ones(x_alpha.shape[0], dtype=th.float32, device=dev)
                lstm_states = _make_lstm_states(policy, x_alpha.shape[0], dev)
                dist_alpha = _get_action_dist(
                    x_alpha, lstm_states=lstm_states, ep_starts=ep_starts
                )
            else:
                dist_alpha = _get_action_dist(x_alpha)

            # Objective: sum of log-probs of fixed target actions a*
            logp = dist_alpha.log_prob(a_star)  # [B]
            obj = logp.sum()
            grads = th.autograd.grad(
                obj, x_alpha, retain_graph=False, create_graph=False
            )[0]
            total_grad += grads

        ig_batch = (xb - B) * (total_grad / m_steps)  # [B, N]
        IG_abs_sum += ig_batch.abs().sum(dim=0)
        total_rows += (end - start)

        if progress:
            print(f"  processed {end}/{R}", end="\r")
    if progress:
        print()

    mean_abs_ig = (IG_abs_sum / max(1, total_rows)).detach().cpu().numpy()
    return mean_abs_ig.astype(np.float32)


def ig_value_attr_bp(model, X: np.ndarray, baseline: str = "mean", baseline_vec: Optional[np.ndarray] = None,
                  m_steps: int = 50, batch_size: int = 256, device: Optional[str] = None, progress: bool = True) -> np.ndarray:
    """
    ACTION-focused Integrated Gradients (IG) for the policy:
      objective at each interpolation point x_alpha is sum(log π(a* | x_alpha)),
      where a* is the policy's deterministic action at the endpoint x.

    Returns mean|IG| per feature over rows in X: [N]
    Works for PPO(MlpPolicy) and RecurrentPPO(MlpLstmPolicy).
    Requires helpers: `_is_recurrent(policy)` and `_make_lstm_states(policy, batch, device)`.
    """
    policy = model.policy
    policy.eval()
    dev = th.device(device or policy.device)
    Xf = th.tensor(X, dtype=th.float32, device=dev)
    R, N = Xf.shape

    # ---- baseline vector ----
    if baseline == "zeros":
        b = th.zeros(N, device=dev)
    elif baseline == "mean":
        b = Xf.mean(dim=0)
    elif baseline == "min":
        b = Xf.min(dim=0).values
    elif baseline == "custom":
        assert baseline_vec is not None and baseline_vec.shape == (N,)
        b = th.tensor(baseline_vec, dtype=th.float32, device=dev)
    else:
        raise ValueError("baseline must be 'zeros'|'mean'|'min'|'custom'")

    alphas = th.linspace(1.0 / m_steps, 1.0, m_steps, device=dev)
    IG_abs_sum = th.zeros(N, dtype=th.float32, device=dev)
    total_rows = 0
    recurrent = _is_recurrent(policy)

    if progress:
        print(f"[IG(action)] rows={R}, m_steps={m_steps}, batch={batch_size}, baseline={baseline}, recurrent={recurrent}")

    # ---- tiny helper to get distribution (works for MLP and Recurrent) ----
    def _get_action_dist(obs_t: th.Tensor, lstm_states=None, ep_starts=None):
        # Try fast path if available (SB3 2.x MLP)
        try:
            if not recurrent:
                return policy.get_distribution(obs_t)
        except Exception:
            pass
        # Manual path
        features = policy.extract_features(obs_t)
        if recurrent:
            latent_pi, _ = policy._process_sequence(features, lstm_states.pi, ep_starts, policy.lstm_actor)
            dist = policy._get_action_dist_from_latent(latent_pi)
        else:
            latent_pi, _ = policy.mlp_extractor(features)
            dist = policy._get_action_dist_from_latent(latent_pi)
        return dist

    for start in range(0, R, batch_size):
        end = min(start + batch_size, R)
        xb = Xf[start:end]  # [B, N]
        B = b.unsqueeze(0).expand_as(xb)

        # ---- build fixed a* (deterministic action at endpoint xb) ----
        with th.no_grad():
            if recurrent:
                ep_starts_b = th.ones(xb.shape[0], dtype=th.float32, device=dev)
                lstm_b = _make_lstm_states(policy, xb.shape[0], dev)
                dist_b = _get_action_dist(xb, lstm_states=lstm_b, ep_starts=ep_starts_b)
            else:
                dist_b = _get_action_dist(xb)
            a_star = dist_b.get_actions(deterministic=True)  # torch tensor

        total_grad = th.zeros_like(xb)

        # ---- IG accumulation along the path ----
        for a in alphas:
            x_alpha = (B + a * (xb - B)).clone().detach().requires_grad_(True)
            if recurrent:
                ep_starts = th.ones(x_alpha.shape[0], dtype=th.float32, device=dev)  # float mask avoids old sb3 bool bug
                lstm_states = _make_lstm_states(policy, x_alpha.shape[0], dev)
                dist_alpha = _get_action_dist(x_alpha, lstm_states=lstm_states, ep_starts=ep_starts)
            else:
                dist_alpha = _get_action_dist(x_alpha)

            # Objective: sum of log-probs of fixed target actions a*
            logp = dist_alpha.log_prob(a_star)  # [B]
            obj = logp.sum()
            grads = th.autograd.grad(obj, x_alpha, retain_graph=False, create_graph=False)[0]
            total_grad += grads

        ig_batch = (xb - B) * (total_grad / m_steps)  # [B, N]
        IG_abs_sum += ig_batch.abs().sum(dim=0)
        total_rows += (end - start)

        if progress:
            print(f"  processed {end}/{R}", end="\r")
    if progress:
        print()

    mean_abs_ig = (IG_abs_sum / max(1, total_rows)).detach().cpu().numpy()
    return mean_abs_ig.astype(np.float32)


# --------------- Orchestrator & plotting ---------------

def compare_spca_corr_ig(
    model,
    reservoir_path: str,                      # prefer fcorr_reservoir_final.npz (has X and y)
    FEATURE_NAMES: Optional[Sequence[str]] = FEATURE_NAMES,
    spca_components: int = 3,
    spca_alpha: float = 1.0,
    corr_name: str = "advantage/return",      # label for printing
    ig_baseline: str = "mean",
    ig_steps: int = 50,
    ig_batch: int = 256,
    top_k: int = 20,
    sample_size: Optional[int] = None,        # subsample rows for speed
    seed: int = 0xC0FFEE,
) -> Dict[str, pd.DataFrame]:
    # load
    X, y, names_saved, meta = load_reservoir(reservoir_path)
    R, N = X.shape

    # choose names
    if FEATURE_NAMES is not None and len(FEATURE_NAMES) == N:
        names = np.array(FEATURE_NAMES, dtype=str)
    else:
        names = names_saved

    # optional subsample for speed (consistent across all three metrics)
    if sample_size is not None and sample_size < R:
        rng = np.random.default_rng(seed)
        idx_rows = rng.choice(R, size=sample_size, replace=False)
        X_use = X[idx_rows]
        y_use = y[idx_rows] if y is not None else None
    else:
        idx_rows = np.arange(R)
        X_use, y_use = X, y

    # SPCA weights
    w_spca, comps = spca_weights_backup(X_use, n_components=spca_components, alpha=spca_alpha)

    # Correlations (if y available)
    r = corr_feature_target(X_use, y_use) if y_use is not None else None

    # IG (critic value)
    ig = ig_value_attr(model, X_use, baseline=ig_baseline, m_steps=ig_steps, batch_size=ig_batch)

    # Build table
    df = pd.DataFrame({"feature": names, "idx": np.arange(N), "spca_w": w_spca, "ig": ig})
    if r is not None:
        df["corr"] = r
        df["abs_corr"] = np.abs(r)

    # Normalize columns for side-by-side comparison (optional)
    df["spca_w_pct"] = df["spca_w"] / max(df["spca_w"].sum(), 1e-12)
    df["ig_pct"] = df["ig"] / max(df["ig"].sum(), 1e-12)
    if r is not None:
        df["abs_corr_pct"] = df["abs_corr"] / max(df["abs_corr"].sum(), 1e-12)

    # Rankings
    df_rank_spca = df.sort_values("spca_w", ascending=False).reset_index(drop=True)
    df_rank_ig   = df.sort_values("ig", ascending=False).reset_index(drop=True)
    if r is not None:
        df_rank_corr = df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    else:
        df_rank_corr = None

    # Print top-k per method
    k = min(top_k, N)
    print(f"[Compare] reservoir={meta['source_file']}, rows_used={len(idx_rows)}, N={N}")
    print(f"Top-{k} by SPCA weight:")
    for i, row in df_rank_spca.head(k).iterrows():
        print(f"  {i+1:>2}. {row['feature']})  w={row['spca_w']:.6f}")
    print(f"\nTop-{k} by IG (value):")
    for i, row in df_rank_ig.head(k).iterrows():
        print(f"  {i+1:>2}. {row['feature']})  ig={row['ig']:.6e}")
    if df_rank_corr is not None:
        print(f"\nTop-{k} by |corr(feature, {corr_name})|:")
        for i, row in df_rank_corr.head(k).iterrows():
            print(f"  {i+1:>2}. {row['feature']})  r={float(row['corr']):+.4f}  |r|={row['abs_corr']:.4f}")

    # Plots (three separate bar charts)
    def _barplot(series, labels, ylabel, title, fname=None, alpha=0.7, color='skyblue'):
        plt.figure(figsize=(10, 3.5))
        vals = series.values
        # use series.index to map back to labels and idx in the same ranking DataFrame
        names_plot = [f"{labels.iloc[i]}" for i in range(len(series))]
        plt.bar(range(len(vals)), vals, color=color, alpha=alpha)
        plt.xticks(range(len(vals)), names_plot, rotation=45, ha="right", fontsize=10)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        if fname:
            plt.savefig(fname, dpi=150)
        plt.show()

    _barplot(df_rank_spca.head(k)["spca_w"], df_rank_spca.head(k)["feature"], "Wagi SPCA", f"Ważność cech wg SPCA", color='black', alpha=0.5)
    if df_rank_corr is not None:
        _barplot(df_rank_corr.head(k)["abs_corr"], df_rank_corr.head(k)["feature"],
         f"|wartość korelacji|", f"Bezwzględna korelacja Pearsona z przewagą", color='steelblue', alpha=0.6)
    _barplot(df_rank_ig.head(k)["ig"], df_rank_ig.head(k)["feature"], "|IG|", f"Średnia bezwzględna wartość IG (akcje)", color='steelblue', alpha=1.0)

    out = {"table": df, "rank_spca": df_rank_spca, "rank_corr": df_rank_corr, "rank_ig": df_rank_ig, "components": comps, "meta": meta}
    return out



def _plot_reservoir_rankings(
        df_rank_spca,
        df_rank_ig,
        df_rank_corr,
        k_plot: int,
        save_prefix: Optional[str] = None,
    ) -> None:
        """
        Visualize top-k SPCA / |corr| / IG rankings from reservoir analysis.

        Parameters
        ----------
        df_rank_spca : pd.DataFrame
            DataFrame sorted by 'spca_w' descending.
        df_rank_ig : pd.DataFrame
            DataFrame sorted by 'ig' descending.
        df_rank_corr : Optional[pd.DataFrame]
            DataFrame sorted by 'abs_corr' descending, or None if y is missing.
        k_plot : int
            Number of top features to show in each plot (clipped to dataframe length).
        save_prefix : Optional[str]
            If not None, save figures as "<save_prefix>_spca.png", "<save_prefix>_corr.png", "<save_prefix>_ig.png".
        """
        import matplotlib.pyplot as plt

        k_plot = max(1, int(k_plot))

        def _barplot(series, labels, ylabel, title, suffix: Optional[str], alpha=0.7, color="skyblue"):
            if len(series) == 0:
                return
            vals = series.values
            names_plot = labels.values

            plt.figure(figsize=(10, 3.5))
            plt.bar(range(len(vals)), vals, color=color, alpha=alpha)
            plt.xticks(range(len(vals)), names_plot, rotation=45, ha="right", fontsize=10)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.tight_layout()
            if save_prefix is not None and suffix is not None:
                plt.savefig(f"{save_prefix}_{suffix}.png", dpi=150)
            plt.show()

        # SPCA
        k_spca = min(k_plot, len(df_rank_spca))
        _barplot(
            df_rank_spca.head(k_spca)["spca_w"],
            df_rank_spca.head(k_spca)["feature"],
            ylabel="Wagi SPCA",
            title="Ważność cech wg SPCA",
            suffix="spca",
            alpha=0.5,
            color="black",
        )

        # Correlation (if available)
        if df_rank_corr is not None:
            k_corr = min(k_plot, len(df_rank_corr))
            _barplot(
                df_rank_corr.head(k_corr)["abs_corr"],
                df_rank_corr.head(k_corr)["feature"],
                ylabel="|wartość korelacji|",
                title="Bezwzględna korelacja Pearsona z przewagą",
                suffix="corr",
                alpha=0.6,
                color="steelblue",
            )

        # IG
        k_ig = min(k_plot, len(df_rank_ig))
        _barplot(
            df_rank_ig.head(k_ig)["ig"],
            df_rank_ig.head(k_ig)["feature"],
            ylabel="|IG|",
            title="Średnia bezwzględna wartość IG (akcje)",
            suffix="ig",
            alpha=1.0,
            color="steelblue",
        )




import numpy as np
from typing import Tuple, Optional
from sklearn.decomposition import SparsePCA

def _winsorize(Z: np.ndarray, clip: float = 5.0) -> np.ndarray:
    if clip is None:
        return Z
    return np.clip(Z, -clip, clip)

def _zscore_cols_robust(X: np.ndarray, clip: Optional[float] = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # standard z-score (your version is fine too); add optional winsorization
    mu = np.nanmean(X, axis=0, keepdims=True)
    Z = X - mu
    sd = np.nanstd(Z, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Z = Z / sd
    Z = _winsorize(Z, clip)
    return Z, mu.squeeze(0), sd.squeeze(0)

def spca_weights_improved(
    X: np.ndarray,
    n_components: int = 3,
    alpha: float = 1.0,
    ridge_alpha: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-8,
    method: str = "cd",                        # more stable than "lars" for many cases
    random_state: int = 0xC0FFEE,
    # --- robustness knobs ---
    min_nonzero_rate: float = 0.02,            # drop columns active < 2% of rows
    min_prestd: float = 0.0,                   # drop columns with std < threshold BEFORE z-score
    clip_z: Optional[float] = 5.0,             # winsorize z-scores to [-clip, clip]
    # --- aggregation knobs ---
    component_weight: str = "var",             # "var" | "l1code" | "none"
    activity_gamma: float = 0.5,               # down-weight rare features: w *= (nz_rate**gamma)
    normalize_weights: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      w_full:      [N] feature weights mapped back to original feature space
      comps_full:  [K, N] components mapped back (zeros for dropped cols)
    """
    R, N = X.shape
    # --- activity/variance gates on RAW X ---
    nz_rate = (np.abs(X) > 0).mean(axis=0)        # [N]
    ok_nz = nz_rate >= float(min_nonzero_rate)
    prestd = np.std(X, axis=0)
    ok_std = prestd >= float(min_prestd)
    keep = ok_nz & ok_std
    if not np.any(keep):
        # nothing passes; fall back to uniform tiny weights
        return np.ones(N, dtype=np.float32) / N, np.zeros((min(n_components, N), N), dtype=np.float32)

    Xk = X[:, keep]
    Zk, _, _ = _zscore_cols_robust(Xk, clip=clip_z)

    Kmax = min(n_components, Zk.shape[0], Zk.shape[1])
    if Kmax <= 0:
        return np.ones(N, dtype=np.float32) / N, np.zeros((min(n_components, N), N), dtype=np.float32)

    spca = SparsePCA(
        n_components=Kmax, alpha=alpha, ridge_alpha=ridge_alpha,
        max_iter=max_iter, tol=tol, method=method, random_state=random_state
    )
    spca.fit(Zk)
    C = spca.components_.astype(np.float64, copy=False)     # [K, Nk]

    # Get component strengths from codes on this dataset
    T = spca.transform(Zk).astype(np.float64, copy=False)   # [R, K]
    if component_weight == "var":
        w_k = np.var(T, axis=0, ddof=1)                     # [K]
    elif component_weight == "l1code":
        w_k = np.mean(np.abs(T), axis=0)                    # [K]
    else:
        w_k = np.ones(C.shape[0], dtype=np.float64)

    # Aggregate per-feature: sum_k |C[k,j]| * w_k
    w_keep = (np.abs(C) * w_k[:, None]).sum(axis=0)         # [Nk]

    # Activity penalty for rare features (in raw X)
    if activity_gamma is not None and activity_gamma != 0.0:
        w_keep = w_keep * np.power(nz_rate[keep] + 1e-12, float(activity_gamma))

    # Map back to full N
    w_full = np.zeros(N, dtype=np.float64)
    w_full[keep] = w_keep
    if normalize_weights and w_full.sum() > 0:
        w_full /= w_full.sum()

    # Map components back to full N for convenience (zeros where dropped)
    comps_full = np.zeros((C.shape[0], N), dtype=np.float64)
    comps_full[:, keep] = C

    return w_full.astype(np.float32), comps_full.astype(np.float32)
