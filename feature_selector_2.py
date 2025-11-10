# feature_selector.py

import os
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import SparsePCA
import stable_baselines3
import sb3_contrib

from utils import FEATURE_NAMES


models = {
    "PPO": {"MlpPolicy"},
    "RecurrentPPO": {"MlpLstmPolicy"},
}

selection_methods = ["attention", "correlation", "spca", "ig"]


class FeatureSelectionTrainer:
    """
    Trainer + feature-selection helper using:
      - Attention logs
      - SPCA / correlation / IG on a reservoir .npz
    """

    def __init__(
        self,
        args: Optional[Any] = None,
        env: Optional[Any] = None,
        selection_method: str = "attention",
        n_features_to_keep: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = FEATURE_NAMES,
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

        callbacks: List[Any] = []

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
                    reservoir_size=70_000,
                    rank_source="contrib",
                    mask_source="contrib",
                    save_npz_path=f"logs/spca_corr_attn_all_{self.args.model_name}/attn_cumulative_final.npz",
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
                    reservoir_size=70_000,
                    n_components=3,
                    alpha=1.0,
                    ridge_alpha=0.01,
                    max_iter=1000,
                    tol=1e-8,
                    method="lars",
                    weight_norm="l1",
                    normalize_weights=True,
                    feature_names=[f"f{i}" for i in range(self.env.observation_space.shape[-1])],
                    save_dir=f"./logs/spca_corr_attn_all_{self.args.model_name}/",
                    tensorboard=True,
                    verbose=1,
                )

            if "correlation" in self.selection_method:
                cb_corr = FeatureCorrelationFromRolloutCallback(
                    compute_every_rollouts=1,
                    warmup_rollouts=2,
                    print_every_steps=50_000,
                    print_top_k=7,
                    redundancy_threshold=0.95,
                    reservoir_size=70_000,
                    target_kind="return",  # or "return" / "reward"
                    feature_names=[f"f{i}" for i in range(self.env.observation_space.shape[-1])],
                    save_dir=f"./logs/spca_corr_attn_all_{self.args.model_name}/",
                    tensorboard=True,
                    verbose=1,
                )

            callbacks = [cb for cb in (cb_spca, cb_corr) if cb is not None]

        # Train
        self.model.learn(total_timesteps=self.args.num_timesteps, callback=callbacks)


    def get_model(self):
        return self.model
    # ------------------------------------------------------------------
    # Public selection entrypoint
    # ------------------------------------------------------------------

    def select_features(self):
        """
        Run feature selection after training.
        - If selection_method == "attention", reads attention .npz
        - Otherwise uses reservoir + SPCA / corr / IG
        """
        selected_indices = None
        selected_names = None

        if self.selection_method == "attention":
            result = self.select_from_attn_npz(
                npz_path=f"logs/spca_corr_attn_all_{self.args.model_name}/attn_cumulative_final.npz",
                source="contrib",
                mode="cumulative",
                tau=0.9,
                k=7,
                bar_value="mean",
                plot=True,
                save_path="logs/topk_contrib_mean.png",
                show=False,
            )
            selected_indices = result["indices"]
            selected_names = result["feature_names"]
        else:
            reservoir_path = (
                f"./logs/spca_corr_attn_all_{self.args.model_name}/fcorr_reservoir_final.npz"
            )
            result = self.select_from_reservoir(
                reservoir_path=reservoir_path,
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
        min_k: int = 3,  # default lower bound
        max_k: Optional[int] = None,
        return_mask: bool = False,
        return_details: bool = False,
    ):
        """
        Select the smallest set of top features whose cumulative score >= tau.
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

        # Transform scores (optional flatten/sharpen)
        scores = self._transform_scores(
            scores_raw,
            transform="log",
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

        if plot:
            if bar_value == "freq":
                vals = final_freq
                ylabel = "Średnia ważność"
                title = f"Ważność cech wg Attention (freq)"
            else:
                vals = final_mean
                ylabel = "Średnia ważność"
                title = f"Średnia ważność cech wg Attention"

            # plot *all* features in the chosen order, not only the selected ones
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

    # ------------------------------------------------------------------
    # Reservoir helpers (SPCA / corr / IG)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_reservoir(path: str):
        d = np.load(path, allow_pickle=False)
        X = d["X"].astype(np.float32)
        y = d["y"].astype(np.float32) if "y" in d.files else None
        if "feature_names" in d.files:
            names = d["feature_names"].astype(str)
        else:
            names = np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)
        meta = {
            "rows_filled": int(d["rows_filled"]) if "rows_filled" in d.files else None,
            "rows_seen": int(d["rows_seen"]) if "rows_seen" in d.files else None,
            "step": int(d["step"]) if "step" in d.files else None,
            "source_file": os.path.basename(path),
        }
        return X, y, names, meta

    # --------- NEW robust SPCA helpers (your implementation) ---------

    @staticmethod
    def _winsorize(Z: np.ndarray, clip: float = 5.0) -> np.ndarray:
        if clip is None:
            return Z
        return np.clip(Z, -clip, clip)

    @staticmethod
    def _zscore_cols_robust(
        X: np.ndarray, clip: Optional[float] = 5.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Robust z-score with optional winsorization (as provided).
        """
        mu = np.nanmean(X, axis=0, keepdims=True)
        Z = X - mu
        sd = np.nanstd(Z, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = Z / sd
        Z = FeatureSelectionTrainer._winsorize(Z, clip)
        return Z, mu.squeeze(0), sd.squeeze(0)
    @staticmethod
    def spca_weights_improved(
        X: np.ndarray,
        n_components: int = 8,               # upper bound on number of components
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        method: str = "cd",                  # more stable than "lars" in many cases
        random_state: int = 0xC0FFEE,
        # --- robustness knobs ---
        min_nonzero_rate: float = 0.02,      # drop columns active < 2% of rows
        min_prestd: float = 0.0,             # drop columns with std < threshold BEFORE z-score
        clip_z: Optional[float] = 5.0,       # winsorize z-scores to [-clip, clip]
        # --- aggregation knobs ---
        component_weight: str = "var",       # "var" | "l1code" | "none"
        activity_gamma: float = 0.5,         # down-weight rare features: w *= (nz_rate**gamma)
        normalize_weights: bool = True,
        # --- NEW: automatic component selection ---
        var_threshold: Optional[float] = 0.9 # keep smallest #components explaining this frac of code variance
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        w_full:      [N] feature weights mapped back to original feature space
        comps_full:  [K_eff, N] effective components mapped back (zeros for dropped cols)

        n_components is treated as a *maximum*. If var_threshold is not None and we use
        variance-based component weights, we keep only as many components as needed to
        explain `var_threshold` of total variance in the SPCA codes.
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
            return (np.ones(N, dtype=np.float32) / N,
                    np.zeros((min(n_components, N), N), dtype=np.float32))

        Xk = X[:, keep]
        Zk, _, _ = FeatureSelectionTrainer._zscore_cols_robust(Xk, clip=clip_z)

        # upper bound on components
        Kmax = min(n_components, Zk.shape[0], Zk.shape[1])
        if Kmax <= 0:
            return (np.ones(N, dtype=np.float32) / N,
                    np.zeros((min(n_components, N), N), dtype=np.float32))

        spca = SparsePCA(
            n_components=Kmax,
            alpha=alpha,
            ridge_alpha=ridge_alpha,
            max_iter=max_iter,
            tol=tol,
            method=method,
            random_state=random_state,
        )
        spca.fit(Zk)
        C = spca.components_.astype(np.float64, copy=False)     # [Kmax, Nk]

        # codes on this dataset
        T = spca.transform(Zk).astype(np.float64, copy=False)   # [R, Kmax]

        # component "strengths"
        if component_weight == "var":
            w_k = np.var(T, axis=0, ddof=1)                     # [Kmax]
        elif component_weight == "l1code":
            w_k = np.mean(np.abs(T), axis=0)                    # [Kmax]
        else:
            w_k = np.ones(C.shape[0], dtype=np.float64)

        # ---------- NEW: choose effective #components by variance threshold ----------
        if (component_weight == "var") and (var_threshold is not None):
            vt = float(var_threshold)
            vt = max(0.0, min(1.0, vt))
            total_var = w_k.sum()
            if total_var > 0 and vt < 1.0:
                order_k = np.argsort(w_k)[::-1]       # best components first
                cum = np.cumsum(w_k[order_k]) / total_var
                K_eff = int(np.searchsorted(cum, vt) + 1)
                K_eff = max(1, min(K_eff, Kmax))

                use_idx = order_k[:K_eff]
                C = C[use_idx]                         # [K_eff, Nk]
                w_k = w_k[use_idx]                     # [K_eff]
            # else: either vt==1 or total_var==0 -> we just use all Kmax components
        # ---------------------------------------------------------------------------

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
    @staticmethod
    def _spca_weights_improved_bp(
        X: np.ndarray,
        n_components: int = 3,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        method: str = "cd",  # more stable than "lars"
        random_state: int = 0xC0FFEE,
        # robustness knobs:
        min_nonzero_rate: float = 0.02,
        min_prestd: float = 0.0,
        clip_z: Optional[float] = 5.0,
        # aggregation knobs:
        component_weight: str = "var",  # "var" | "l1code" | "none"
        activity_gamma: float = 0.5,  # down-weight rare features
        normalize_weights: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robust SPCA feature weights (your improved implementation).

        Returns:
          w_full:      [N] feature weights mapped back to original feature space
          comps_full:  [K, N] components mapped back (zeros for dropped cols)
        """
        R, N = X.shape

        # activity / variance gates on RAW X
        nz_rate = (np.abs(X) > 0).mean(axis=0)  # [N]
        ok_nz = nz_rate >= float(min_nonzero_rate)
        prestd = np.std(X, axis=0)
        ok_std = prestd >= float(min_prestd)
        keep = ok_nz & ok_std

        if not np.any(keep):
            # nothing passes; fall back to uniform tiny weights
            return (
                np.ones(N, dtype=np.float32) / N,
                np.zeros((min(n_components, N), N), dtype=np.float32),
            )

        Xk = X[:, keep]
        Zk, _, _ = FeatureSelectionTrainer._zscore_cols_robust(Xk, clip=clip_z)

        Kmax = min(n_components, Zk.shape[0], Zk.shape[1])
        if Kmax <= 0:
            return (
                np.ones(N, dtype=np.float32) / N,
                np.zeros((min(n_components, N), N), dtype=np.float32),
            )

        spca = SparsePCA(
            n_components=Kmax,
            alpha=alpha,
            ridge_alpha=ridge_alpha,
            max_iter=max_iter,
            tol=tol,
            method=method,
            random_state=random_state,
        )
        spca.fit(Zk)
        C = spca.components_.astype(np.float64, copy=False)  # [K, Nk]

        # codes on this dataset
        T = spca.transform(Zk).astype(np.float64, copy=False)  # [R, K]
        if component_weight == "var":
            w_k = np.var(T, axis=0, ddof=1)  # [K]
        elif component_weight == "l1code":
            w_k = np.mean(np.abs(T), axis=0)
        else:
            w_k = np.ones(C.shape[0], dtype=np.float64)

        # Aggregate per-feature: sum_k |C[k,j]| * w_k
        w_keep = (np.abs(C) * w_k[:, None]).sum(axis=0)  # [Nk]

        # Activity penalty for rare features (in raw X)
        if activity_gamma is not None and activity_gamma != 0.0:
            w_keep = w_keep * np.power(nz_rate[keep] + 1e-12, float(activity_gamma))

        # Map back to full N
        w_full = np.zeros(N, dtype=np.float64)
        w_full[keep] = w_keep
        if normalize_weights and w_full.sum() > 0:
            w_full /= w_full.sum()

        # Map components back to full N
        comps_full = np.zeros((C.shape[0], N), dtype=np.float64)
        comps_full[:, keep] = C

        return w_full.astype(np.float32), comps_full.astype(np.float32)

    # ------------------------------------------------------------------
    # Correlation + IG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _corr_feature_target(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Pearson correlation r_j = corr(X[:, j], y)
        """
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

    @staticmethod
    def _is_recurrent(policy) -> bool:
        return hasattr(policy, "lstm_actor") and hasattr(policy, "lstm_critic")

    @staticmethod
    @th.no_grad()
    def _make_lstm_states(policy, batch: int, device: th.device):
        """
        Build LSTM states in the structure expected by sb3-contrib.
        """
        la, lc = policy.lstm_actor, policy.lstm_critic
        h_a = th.zeros(la.num_layers, batch, la.hidden_size, device=device)
        c_a = th.zeros_like(h_a)
        h_c = th.zeros(lc.num_layers, batch, lc.hidden_size, device=device)
        c_c = th.zeros_like(h_c)
        try:
            from sb3_contrib.common.recurrent.type_aliases import RecurrentStates

            return RecurrentStates(pi=(h_a, c_a), vf=(h_c, c_c))
        except Exception:
            from types import SimpleNamespace

            return SimpleNamespace(pi=(h_a, c_a), vf=(h_c, c_c))

    def _ig_value_attr(
        self,
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
        """
        if self.model is None:
            raise RuntimeError("Model is not set. Call train_model() first.")

        policy = self.model.policy
        policy.eval()
        dev = th.device(device or policy.device)
        Xf = th.tensor(X, dtype=th.float32, device=dev)
        R, N = Xf.shape

        # baseline vector
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
        recurrent = self._is_recurrent(policy)

        if progress:
            print(
                f"[IG(action)] rows={R}, m_steps={m_steps}, batch={batch_size}, "
                f"baseline={baseline}, recurrent={recurrent}"
            )

        def _get_action_dist(obs_t: th.Tensor, lstm_states=None, ep_starts=None):
            """
            Return a distribution object for actions given obs_t.
            Works for MlpPolicy and RecurrentPPO(MlpLstmPolicy).
            """
            if not recurrent:
                try:
                    return policy.get_distribution(obs_t)
                except Exception:
                    features = policy.extract_features(obs_t)
                    latent_pi, _ = policy.mlp_extractor(features)
                    return policy._get_action_dist_from_latent(latent_pi)

            features = policy.extract_features(obs_t)

            if lstm_states is None:
                lstm_states = self._make_lstm_states(policy, obs_t.shape[0], dev)
            if ep_starts is None:
                ep_starts = th.ones(obs_t.shape[0], dtype=th.float32, device=dev)

            lstm_out_pi, _ = policy._process_sequence(
                features, lstm_states.pi, ep_starts, policy.lstm_actor
            )
            latent_pi, _ = policy.mlp_extractor(lstm_out_pi)
            dist = policy._get_action_dist_from_latent(latent_pi)
            return dist

        for start in range(0, R, batch_size):
            end = min(start + batch_size, R)
            xb = Xf[start:end]  # [B, N]
            B = b.unsqueeze(0).expand_as(xb)

            # fixed a* at endpoint xb
            with th.no_grad():
                if recurrent:
                    ep_starts_b = th.ones(xb.shape[0], dtype=th.float32, device=dev)
                    lstm_b = self._make_lstm_states(policy, xb.shape[0], dev)
                    dist_b = _get_action_dist(xb, lstm_states=lstm_b, ep_starts=ep_starts_b)
                else:
                    dist_b = _get_action_dist(xb)
                a_star = dist_b.get_actions(deterministic=True)

            total_grad = th.zeros_like(xb)

            # IG path
            for a in alphas:
                x_alpha = (B + a * (xb - B)).clone().detach().requires_grad_(True)
                if recurrent:
                    ep_starts = th.ones(x_alpha.shape[0], dtype=th.float32, device=dev)
                    lstm_states = self._make_lstm_states(policy, x_alpha.shape[0], dev)
                    dist_alpha = _get_action_dist(
                        x_alpha, lstm_states=lstm_states, ep_starts=ep_starts
                    )
                else:
                    dist_alpha = _get_action_dist(x_alpha)

                logp = dist_alpha.log_prob(a_star)
                obj = logp.sum()
                grads = th.autograd.grad(
                    obj, x_alpha, retain_graph=False, create_graph=False
                )[0]
                total_grad += grads

            ig_batch = (xb - B) * (total_grad / m_steps)
            IG_abs_sum += ig_batch.abs().sum(dim=0)
            total_rows += (end - start)

            if progress:
                print(f"  processed {end}/{R}", end="\r")
        if progress:
            print()

        mean_abs_ig = (IG_abs_sum / max(1, total_rows)).detach().cpu().numpy()
        return mean_abs_ig.astype(np.float32)

    # ------------------------------------------------------------------
    # High-level: feature selection from reservoir (SPCA / corr / IG)
    # ------------------------------------------------------------------

    def select_from_reservoir(
        self,
        reservoir_path: str,
        method: Optional[str] = None,  # "spca" | "correlation" | "ig"
        mode: str = "topk",
        k: int = 20,
        tau: float = 0.90,
        min_k: int = 1,
        max_k: Optional[int] = None,
        spca_components: int = 3,
        spca_alpha: float = 1.0,
        corr_name: str = "advantage/return",
        ig_baseline: str = "mean",
        ig_steps: int = 50,
        ig_batch: int = 256,
        sample_size: Optional[int] = None,
        seed: int = 0xC0FFEE,
        verbose: bool = True,
        plot: bool = False,
        top_k_plot: int = 20,
        save_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run SPCA / correlation / IG feature selection using a reservoir .npz file.
        """
        X, y, names_saved, meta = self._load_reservoir(reservoir_path)
        R, N = X.shape

        if self.feature_names is not None and len(self.feature_names) == N:
            names = np.array(self.feature_names, dtype=str)
        else:
            names = names_saved

        # Subsample for speed (applied consistently across all methods)
        if sample_size is not None and sample_size < R:
            rng = np.random.default_rng(seed)
            idx_rows = rng.choice(R, size=sample_size, replace=False)
            X_use = X[idx_rows]
            y_use = y[idx_rows] if y is not None else None
        else:
            idx_rows = np.arange(R)
            X_use, y_use = X, y

        # --- Robust SPCA weights (your improved implementation) ---
        w_spca, comps = FeatureSelectionTrainer.spca_weights_improved(
            X_use,
            n_components=7,       # upper bound
            alpha=spca_alpha,
            var_threshold=0.9,
        )

        # Correlations (if y available)
        r = self._corr_feature_target(X_use, y_use) if y_use is not None else None

        # IG (action-focused)
        ig = self._ig_value_attr(
            X_use,
            baseline=ig_baseline,
            m_steps=ig_steps,
            batch_size=ig_batch,
        )

        # Build comparison table
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

        # Rankings for diagnostics / plotting
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
                f"[Reservoir] file={meta['source_file']}, rows_used={len(idx_rows)}, N={N}"
            )
            print(f"Top-{k_show} cech wg wagi SPCA:")
            for i, row in df_rank_spca.head(k_show).iterrows():
                print(
                    f"  {i+1:>2}. {row['feature']} (#{int(row['idx'])})  w={row['spca_w']:.6f}"
                )
            print(f"\nTop-{k_show} cech wg IG (akcje):")
            for i, row in df_rank_ig.head(k_show).iterrows():
                print(
                    f"  {i+1:>2}. {row['feature']} (#{int(row['idx'])})  ig={row['ig']:.6e}"
                )
            if df_rank_corr is not None:
                print(f"\nTop-{k_show} cech wg |corr(cecha, {corr_name})|:")
                for i, row in df_rank_corr.head(k_show).iterrows():
                    print(
                        f"  {i+1:>2}. {row['feature']} (#{int(row['idx'])})  "
                        f"r={float(row['corr']):+.4f}  |r|={row['abs_corr']:.4f}"
                    )

        # Choose metric for selection
        m = (method or self.selection_method or "spca").lower()
        assert m in {"spca", "correlation", "ig"}

        if m == "spca":
            scores = df["spca_w"].to_numpy()
        elif m == "ig":
            scores = df["ig"].to_numpy()
        else:
            if r is None:
                raise RuntimeError(
                    "Reservoir does not have y; cannot compute correlation-based scores."
                )
            scores = df["abs_corr"].to_numpy()

        scores = np.asarray(scores, dtype=float)
        order = np.argsort(-scores)

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
                f"\n[Selection] method={m}, mode={mode}, selected {k_sel} features"
            )
            for rank, i in enumerate(order[:k_sel], 1):
                star = "*" if i in selected_idx else " "
                print(
                    f"{rank:>2}{star} {names[i]:<20} idx={i:<4}  score={scores[i]:.6g}"
                )

        if self.feature_names is None:
            self.feature_names = list(names)
        selected_names = [names[i] for i in selected_idx]

        # Optional visualization
        if plot:
            self._plot_reservoir_rankings(
                df_rank_spca=df_rank_spca,
                df_rank_corr=df_rank_corr,
                df_rank_ig=df_rank_ig,
                top_k=min(top_k_plot, N),
                save_prefix=save_prefix,
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

    @staticmethod
    def _plot_reservoir_rankings(
        df_rank_spca: pd.DataFrame,
        df_rank_corr: Optional[pd.DataFrame],
        df_rank_ig: pd.DataFrame,
        top_k: int = 20,
        save_prefix: Optional[str] = None,
    ) -> None:
        """
        Plot top-k features by SPCA weight, |corr| (if available) and IG.
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

        if df_rank_corr is not None:
            k_corr = min(top_k, len(df_rank_corr))
            _barplot(
                df_rank_corr.head(k_corr)["abs_corr"],
                df_rank_corr.head(k_corr)["feature"],
                "|corr|",
                f"Top {k_corr} wg |korelacji|",
                fname=f"{save_prefix}_corr.png" if save_prefix else None,
                color="steelblue",
                alpha=0.6,
            )

        k_ig = min(top_k, len(df_rank_ig))
        _barplot(
            df_rank_ig.head(k_ig)["ig"],
            df_rank_ig.head(k_ig)["feature"],
            "|IG| (akacje)",
            f"Średnia bezwzględna ważność cech wg IG (akcje)",
            fname=f"{save_prefix}_ig.png" if save_prefix else None,
            color="steelblue",
            alpha=1.0,
        )

    # ------------------------------------------------------------------
    # Score transform helper (for attention)
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_scores(
        scores,
        transform: str = "none",  # "none" | "power" | "log" | "sqrt"
        alpha: float = 1.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Apply a monotonic transform to scores before selection.
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
    
    def select_by_correlation_bootstrap(
        self,
        reservoir_path: str,
        k: int = 3,
        B: int = 100,
        sample_frac: float = 0.7,
        corr_name: str = "advantage/return",
        seed: int = 0xC0FFEE,
        mode: str = "topk",          # "topk" or "cumulative"
        tau: float = 0.9,            # used when mode="cumulative"
        min_k: int = 1,
        max_k: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Correlation-based feature selection with bootstrap stabilization.

        Two selection modes:
          - mode="topk":       pick exactly k features by stable bootstrap ranking
          - mode="cumulative": pick the smallest set whose cumulative score >= tau
                                using mean_abs_corr as scores (via select_indices_by_cumulative).
        """
        # 1) Load reservoir
        X, y, names_saved, meta = self._load_reservoir(reservoir_path)
        if y is None:
            raise RuntimeError(
                f"Reservoir at '{reservoir_path}' does not contain 'y'; "
                f"cannot compute correlation-based importance."
            )
        R, N = X.shape

        # Choose feature names (prefer external if consistent)
        if self.feature_names is not None and len(self.feature_names) == N:
            names = np.array(self.feature_names, dtype=str)
        else:
            names = names_saved

        # 2) Bootstrap settings
        B = int(B)
        if B <= 0:
            raise ValueError("B (number of bootstraps) must be >= 1.")
        sample_frac = float(sample_frac)
        if not (0.0 < sample_frac <= 1.0):
            raise ValueError("sample_frac must be in (0,1].")

        rng = np.random.default_rng(seed)
        sample_size = max(1, int(round(sample_frac * R)))

        # Storage for bootstrap stats
        abs_corr_boot = np.zeros((B, N), dtype=float)
        ranks_boot = np.zeros((B, N), dtype=float)

        # 3) Bootstrap loop
        for b in range(B):
            idx = rng.integers(0, R, size=sample_size, endpoint=False)
            Xb = X[idx]
            yb = y[idx]

            r_b = self._corr_feature_target(Xb, yb)   # [N], signed corr
            r_abs = np.abs(r_b).astype(float)         # [N]

            abs_corr_boot[b] = r_abs

            # Convert to ranks (1 = best)
            order_b = np.argsort(-r_abs)              # best first
            ranks_b = np.empty(N, dtype=int)
            ranks_b[order_b] = np.arange(1, N + 1)    # 1..N
            ranks_boot[b] = ranks_b

        # 4) Aggregate over bootstraps
        mean_abs_corr = abs_corr_boot.mean(axis=0)     # [N]
        mean_rank = ranks_boot.mean(axis=0)            # [N]
        topk_freq = (ranks_boot <= k).mean(axis=0)     # [N], in [0,1]

        # Stable ordering: primarily by mean_rank, then by mean_abs_corr
        # np.lexsort: last row is primary key
        order_stable = np.lexsort((-mean_abs_corr, mean_rank))  # [N]

        # 5) Decide selection mode
        mode = str(mode).lower()
        if mode == "cumulative":
            # Use mean_abs_corr as scores for cumulative coverage
            scores = mean_abs_corr.copy()

            selected_idx, details = self.select_indices_by_cumulative(
                scores=scores,
                tau=tau,
                normalize=True,
                min_k=min_k,
                max_k=max_k,
                return_details=True,
            )
            k_sel = selected_idx.size

        else:  # "topk"
            k_sel = min(k, N)
            selected_idx = order_stable[:k_sel]

        selected_names = [names[i] for i in selected_idx]

        # 6) Build DataFrame with all stats
        import pandas as pd
        df = pd.DataFrame(
            {
                "feature": names,
                "idx": np.arange(N),
                "mean_abs_corr": mean_abs_corr,
                "mean_rank": mean_rank,
                "topk_freq": topk_freq,
            }
        )

        # For convenience, also provide a version sorted by the stable order
        df_sorted = df.iloc[order_stable].reset_index(drop=True)

        if verbose:
            if mode == "cumulative":
                print(
                    f"[CorrBootstrap] reservoir={meta.get('source_file', '?')}, "
                    f"rows={R}, features={N}, B={B}, sample_frac={sample_frac:.2f}, "
                    f"mode='cumulative', tau={tau:.3f}, min_k={min_k}, max_k={max_k}"
                )
            else:
                print(
                    f"[CorrBootstrap] reservoir={meta.get('source_file', '?')}, "
                    f"rows={R}, features={N}, B={B}, sample_frac={sample_frac:.2f}, "
                    f"mode='topk', k={k_sel}"
                )

            print(f"Target: {corr_name}")
            for r, i in enumerate(order_stable[:k_sel], 1):
                flag = "*" if i in selected_idx else " "
                print(
                    f"  {r:>2}{flag} {names[i]:<25} "
                    f"idx={i:<3d}  mean|corr|={mean_abs_corr[i]:.4f}  "
                    f"mean_rank={mean_rank[i]:.2f}  topk_freq={topk_freq[i]:.2f}"
                )

        # Sync feature_names if not set
        if self.feature_names is None:
            self.feature_names = list(names)

        return {
            "indices": selected_idx,
            "feature_names": selected_names,
            "mean_abs_corr": mean_abs_corr,
            "mean_rank": mean_rank,
            "topk_freq": topk_freq,
            "order": order_stable,
            "df": df_sorted,
            "meta": meta,
        }
