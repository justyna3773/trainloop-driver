# feature_correlation_callback.py
# feature_correlation_callback.py
from typing import Optional, List
import os
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

class FeatureCorrelationFromRolloutCallback(BaseCallback):
    """
    Correlation diagnostics for (Recurrent)PPO with MlpLstmPolicy.

    At selected rollout ends (no extra env steps):
      • Pull rollout observations [T, B, N] and a target vector (advantages/returns/rewards)
      • Add rows to a fixed-size reservoir (uniform sample of all rows seen so far)
      • At print cadence -> compute & print:
          - feature->target Pearson r  (ranked by |r|)
          - feature<->feature correlation (flag redundant pairs |rho| >= threshold)
      • Periodic NPZ/CSV dumps if save_dir is set
      • FINAL snapshot saved at training end (_on_training_end)

    Notes:
      - Reservoir bounds memory; attention/cum stats elsewhere are unaffected.
      - Robust to zero-variance columns (returns 0 correlation for them).
    """

    def __init__(
        self,
        compute_every_rollouts: int = 1,           # compute at these rollout intervals (after warmup)
        warmup_rollouts: int = 2,                  # skip first N rollouts
        print_every_steps: Optional[int] = 100_000,# throttle printing/logging by env steps (aligned to rollout end)
        print_top_k: int = 20,                     # how many features to print by |corr|
        redundancy_threshold: float = 0.95,        # flag |rho| >= thr as redundant
        reservoir_size: int = 20_000,              # number of rows kept for correlations (bounded memory)
        target_kind: str = "advantage",            # "advantage" | "return" | "reward"
        feature_names: Optional[List[str]] = None,
        save_dir: Optional[str] = None,            # write NPZ/CSV snapshots here
        tensorboard: bool = True,                  # log summary scalars to TB if model has TB logger
        verbose: int = 1,
        final_basename: str = "final",             # files: fcorr_final.* written on training end
    ):
        super().__init__(verbose)
        assert target_kind in {"advantage", "return", "reward"}
        self.compute_every_rollouts = int(compute_every_rollouts)
        self.warmup_rollouts = int(warmup_rollouts)
        self.print_every_steps = int(print_every_steps) if print_every_steps is not None else None
        self.print_top_k = int(print_top_k)
        self.redundancy_threshold = float(redundancy_threshold)
        self.reservoir_size = int(reservoir_size)
        self.target_kind = target_kind
        self.feature_names = feature_names
        self.save_dir = save_dir
        self.tensorboard = tensorboard
        self.final_basename = str(final_basename)

        # state
        self.rollouts_seen = 0
        self._last_print_ts = 0
        self._rng = np.random.default_rng(0xC0FFEE)
        self._N: Optional[int] = None

        # reservoir buffers
        self._X: Optional[np.ndarray] = None   # [R, N] float32
        self._y: Optional[np.ndarray] = None   # [R]    float32
        self._rows_seen = 0
        self._filled = 0

        # last computed snapshot (for final save)
        self._last_C_ff: Optional[np.ndarray] = None   # [N,N]
        self._last_r_ft: Optional[np.ndarray] = None   # [N]
        self._last_order: Optional[List[int]] = None   # ranked indices by |r|
        self._last_names: Optional[List[str]] = None

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    # SB3 compatibility (some versions mark _on_step abstract)
    def _on_step(self) -> bool:
        return True

    # ------------------- helpers -------------------

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

    def _pick_target(self, buf) -> np.ndarray:
        if self.target_kind == "advantage":
            t = getattr(buf, "advantages", None)
        elif self.target_kind == "return":
            t = getattr(buf, "returns", None)
        else:
            t = getattr(buf, "rewards", None)
        if t is None:
            t = getattr(buf, "advantages", None) or getattr(buf, "returns", None) or getattr(buf, "rewards", None)
        if t is None:
            raise RuntimeError("Could not find a suitable target (advantages/returns/rewards) in rollout buffer.")
        return self._to_np(t)

    def _reservoir_add(self, X_batch: np.ndarray, y_batch: np.ndarray):
        B, N = X_batch.shape
        if self._X is None:
            size = self.reservoir_size if self.reservoir_size > 0 else B
            self._X = np.empty((size, N), dtype=np.float32)
            self._y = np.empty((size,), dtype=np.float32)
            self._filled = 0
            self._rows_seen = 0
        for i in range(B):
            self._rows_seen += 1
            if self.reservoir_size <= 0:
                # unbounded (realloc doubling) – use with care
                if self._filled >= self._X.shape[0]:
                    self._X = np.vstack([self._X, np.empty_like(self._X)])
                    self._y = np.concatenate([self._y, np.empty_like(self._y)])
            if self._filled < self._X.shape[0]:
                dst = self._filled
                self._filled += 1
            else:
                j = self._rng.integers(0, self._rows_seen)
                if j >= self._X.shape[0]:
                    continue
                dst = int(j)
            self._X[dst] = X_batch[i]
            self._y[dst] = y_batch[i]

    @staticmethod
    def _safe_corrcoef(M: np.ndarray) -> np.ndarray:
        """Like np.corrcoef(M, rowvar=False) but robust to zero-variance cols: returns 0 corr."""
        M = M.astype(np.float64, copy=False)
        M = M - np.nanmean(M, axis=0, keepdims=True)
        std = np.nanstd(M, axis=0, keepdims=True)
        std[std == 0] = np.nan
        Z = M / std
        C = np.nan_to_num(Z).T @ np.nan_to_num(Z)
        C /= max(1, Z.shape[0] - 1)
        C = np.clip(C, -1.0, 1.0)
        return C

    @staticmethod
    def _pearson_ft(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Pearson r(feature, target) per column; robust to zero-variance."""
        X = X.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        X = X - np.nanmean(X, axis=0, keepdims=True)
        y = y - np.nanmean(y, axis=0, keepdims=True)
        sx = np.nanstd(X, axis=0)
        sy = float(np.nanstd(y))
        sx[sx == 0] = np.nan
        sy = sy if sy != 0 else np.nan
        r = (X * y[:, None]).sum(axis=0) / ((X.shape[0] - 1) * sx * sy)
        r = np.clip(np.nan_to_num(r, nan=0.0), -1.0, 1.0)
        return r

    def _analyze_and_report(self, step_ts: int, dump_snapshot: bool = True):
        if self._X is None or self._filled < 2:
            return

        X = self._X[: self._filled]          # [R, N]
        y = self._y[: self._filled]          # [R]
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]

        # compute correlations
        C_ff = self._safe_corrcoef(X)        # [N, N]
        r_ft = self._pearson_ft(X, y)        # [N]
        order = np.argsort(-np.abs(r_ft)).tolist()

        # cache last snapshot for final save
        self._last_C_ff = C_ff
        self._last_r_ft = r_ft
        self._last_order = order
        self._last_names = names

        # print summary
        if self.verbose:
            label = f"step={step_ts:,}" if step_ts >= 0 else "final"
            print(f"[FeatureCorr] report @ {label}: reservoir_rows={self._filled:,} (of seen={self._rows_seen:,})  target={self.target_kind}")

            # redundancy
            thr = self.redundancy_threshold
            pairs = []
            N = X.shape[1]
            for i in range(N):
                for j in range(i + 1, N):
                    if abs(C_ff[i, j]) >= thr:
                        pairs.append((i, j, float(C_ff[i, j])))
            if pairs:
                print(f"  redundant pairs (|rho| >= {thr:.2f}): up to 10 examples")
                for i, j, rho in pairs[:10]:
                    print(f"    ({names[i]}, {names[j]})  rho={rho:+.3f}")

            # top-k by |corr(feature, target)|
            k = min(self.print_top_k, len(order))
            print(f"  top-{k} features by |corr(feature, {self.target_kind})|:")
            for rank, idx in enumerate(order[:k], 1):
                print(f"    {rank:>2}. {names[idx]} (idx={idx})  r={float(r_ft[idx]):+.4f}")


        # tensorboard summary scalars
        if self.tensorboard:
            try:
                self.logger.record("fcorr/mean_abs_r_ft", float(np.mean(np.abs(r_ft))))
                off = np.abs(C_ff[~np.eye(C_ff.shape[0], dtype=bool)])
                self.logger.record("fcorr/mean_abs_offdiag_ff", float(off.mean()))
            except Exception:
                pass

        # periodic file dump
        if dump_snapshot and self.save_dir:
            stem = f"step_{step_ts:010d}" if step_ts >= 0 else "final"
            np.savez(
                os.path.join(self.save_dir, f"fcorr_{stem}.npz"),
                feature_feature_corr=C_ff,
                feature_target_corr=r_ft,
                ranked_indices=np.array(order, dtype=np.int64),
                feature_names=np.array(names),
                rows_filled=self._filled,
                rows_seen=self._rows_seen,
                target_kind=self.target_kind,
                step=step_ts,
            )
            try:
                import pandas as pd
                # per-feature r
                pd.DataFrame({"feature": names, "r_ft": r_ft}).to_csv(
                    os.path.join(self.save_dir, f"feature_target_{stem}.csv"), index=False
                )
                # feature-feature corr matrix
                pd.DataFrame(C_ff, index=names, columns=names).to_csv(
                    os.path.join(self.save_dir, f"feature_feature_{stem}.csv")
                )
                # ranked list
                pd.DataFrame({
                    "rank": np.arange(1, len(order) + 1),
                    "idx": order,
                    "feature": [names[i] for i in order],
                    "r_ft": [r_ft[i] for i in order],
                    "abs_r_ft": [abs(r_ft[i]) for i in order],
                }).to_csv(os.path.join(self.save_dir, f"fcorr_ranking_{stem}.csv"), index=False)
            except Exception:
                pass

    # ---------------- SB3 hooks ----------------

    def _on_rollout_end(self) -> bool:
        self.rollouts_seen += 1
        if self.rollouts_seen < self.warmup_rollouts:
            return True
        if (self.rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
            return True

        buf = self.model.rollout_buffer
        obs = self._to_np(buf.observations)    # [T, B, N] (Recurrent)PPO
        if obs.ndim == 2:                      # rare fallback: [T, N] -> [T,1,N]
            obs = obs[:, None, :]
        T, B, N = obs.shape
        if self._N is None:
            self._N = N
            if self.feature_names is not None and len(self.feature_names) != N:
                raise ValueError(f"feature_names length {len(self.feature_names)} != N={N}")

        # choose target
        target = self._pick_target(buf)        # [T, B] or [T] depending on SB3
        if target.ndim == 1:
            target = target[:, None]

        # flatten rows
        X_rows = obs.reshape(-1, N).astype(np.float32)
        y_rows = target.reshape(-1).astype(np.float32)

        # add to reservoir
        if self.reservoir_size == 0:
            # disable reservoir: use just this rollout
            self._X = X_rows.copy()
            self._y = y_rows.copy()
            self._filled = len(y_rows)
            self._rows_seen += len(y_rows)
        else:
            self._reservoir_add(X_rows, y_rows)

        # cadence aligned to rollout ends
        if self.print_every_steps is not None:
            if (self.num_timesteps - self._last_print_ts) >= self.print_every_steps:
                self._analyze_and_report(self.num_timesteps, dump_snapshot=True)
                self._last_print_ts = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        """
        Save a final snapshot of correlations and ranking (uses the current reservoir).
        """
        if self._X is None or self._filled < 2:
            if self.verbose:
                print("[FeatureCorr] training_end: not enough data to save final snapshot.")
            return

        # recompute once to ensure freshest snapshot; step_ts = -1 marks 'final'
        self._analyze_and_report(step_ts=-1, dump_snapshot=False)

        if not self.save_dir:
            if self.verbose:
                print("[FeatureCorr] training_end: save_dir is None, skipping file save.")
            return

        # use cached last snapshot
        C_ff = self._last_C_ff
        r_ft = self._last_r_ft
        order = self._last_order or list(np.argsort(-np.abs(r_ft)).tolist())
        names = self._last_names or [f"f{i}" for i in range(self._X.shape[1])]

        stem = self.final_basename  # e.g., "final"
        np.savez(
            os.path.join(self.save_dir, f"fcorr_{stem}.npz"),
            feature_feature_corr=C_ff,
            feature_target_corr=r_ft,
            ranked_indices=np.array(order, dtype=np.int64),
            feature_names=np.array(names),
            rows_filled=self._filled,
            rows_seen=self._rows_seen,
            target_kind=self.target_kind,
            step=self.num_timesteps,
        )
        try:
            import pandas as pd
            pd.DataFrame({"feature": names, "r_ft": r_ft}).to_csv(
                os.path.join(self.save_dir, f"feature_target_{stem}.csv"), index=False
            )
            pd.DataFrame(C_ff, index=names, columns=names).to_csv(
                os.path.join(self.save_dir, f"feature_feature_{stem}.csv")
            )
            pd.DataFrame({
                "rank": np.arange(1, len(order) + 1),
                "idx": order,
                "feature": [names[i] for i in order],
                "r_ft": [r_ft[i] for i in order],
                "abs_r_ft": [abs(r_ft[i]) for i in order],
            }).to_csv(os.path.join(self.save_dir, f"fcorr_ranking_{stem}.csv"), index=False)
        except Exception:
            pass
        
                # after saving fcorr_final files...
        if self.save_dir and self._X is not None and self._filled > 0:
            names = self._last_names or [f"f{i}" for i in range(self._X.shape[1])]
            np.savez(
                os.path.join(self.save_dir, "fcorr_reservoir_final.npz"),
                X=self._X[: self._filled],                  # [R, N]
                y=self._y[: self._filled] if self._y is not None else np.array([]),
                feature_names=np.array(names),
                rows_filled=self._filled,
                rows_seen=self._rows_seen,
                step=self.num_timesteps,
            )
            if self.verbose:
                print(f"[FeatureCorr] saved reservoir to {os.path.join(self.save_dir, 'fcorr_reservoir_final.npz')}")

        if self.verbose:
            print(f"[FeatureCorr] saved final correlation snapshot to {self.save_dir} (basename='{stem}')")



# from sb3_contrib import RecurrentPPO


# cb_corr = FeatureCorrelationFromRolloutCallback(
#     compute_every_rollouts=1,          # update every rollout
#     warmup_rollouts=2,
#     print_every_steps=100_000,         # print/log every ~100k env steps
#     print_top_k=20,
#     redundancy_threshold=0.95,
#     reservoir_size=20_000,
#     target_kind="advantage",           # or "return" / "reward"
#     feature_names=[f"f{i}" for i in range(env.observation_space.shape[-1])],
#     save_dir="logs/fcorr_run1",        # set None to disable file dumps
#     tensorboard=True,
#     verbose=1,
# )

# model.learn(total_timesteps=1_000_000, callback=cb_corr)

# feature_spca_callback_strict.py
# feature_spca_callback_strict.py
from typing import Optional, List, Tuple
import os
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.decomposition import SparsePCA  # hard requirement

class FeatureSPCAFromRolloutCallback(BaseCallback):
    """
    Compute per-feature SparsePCA weights from top-K components on rollout data.
    Saves a final snapshot (components, per-feature weights, and ranked list) at training end.
    """

    def __init__(
        self,
        # rollout cadence
        compute_every_rollouts: int = 1,
        warmup_rollouts: int = 2,
        # reporting cadence
        print_every_steps: Optional[int] = 100_000,
        print_top_k: int = 20,
        # reservoir
        reservoir_size: int = 20_000,
        # SPCA config
        n_components: int = 3,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        method: str = "lars",
        weight_norm: str = "l1",           # "l1" (sum |load|) or "l2" (sqrt sum load^2)
        normalize_weights: bool = True,    # normalize weights to sum=1
        random_state: int = 0xC0FFEE,
        # IO / logging
        feature_names: Optional[List[str]] = None,
        save_dir='./logs',    # write NPZ/CSV snapshots here
        tensorboard: bool = True,
        verbose: int = 1,
        # final save controls
        final_basename: str = "final",     # files: spca_final.{npz,csv}, spca_weights_final.csv, etc.
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

        # state
        self.rollouts_seen = 0
        self._last_print_ts = 0
        self._rng = np.random.default_rng(self.random_state)
        self._N: Optional[int] = None

        # reservoir
        self._X: Optional[np.ndarray] = None   # [R, N]
        self._rows_seen = 0
        self._filled = 0

        # last computed snapshot (for final save)
        self._last_components: Optional[np.ndarray] = None   # [K, N]
        self._last_weights: Optional[np.ndarray] = None      # [N]
        self._last_ranked_idx: Optional[List[int]] = None    # length N (best→worst)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    # SB3 compatibility
    def _on_step(self) -> bool:
        return True

    # -------------------- helpers --------------------

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
    def _zscore(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize columns; zero-variance columns get std=1 (become 0 after centering)."""
        mu = np.nanmean(X, axis=0, keepdims=True)
        Z = X - mu
        sd = np.nanstd(Z, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        return Z / sd, mu.squeeze(0), sd.squeeze(0)

    def _fit_spca(self, X: np.ndarray) -> np.ndarray:
        """
        Fit SparsePCA on standardized X and return components [K, N].
        """
        from sklearn.decomposition import SparsePCA

        K = min(self.n_components, X.shape[1])
        Z, _, _ = self._zscore(X)

        spca = SparsePCA(
            n_components=K,
            alpha=self.alpha,
            ridge_alpha=self.ridge_alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            method=self.method,          # "lars" or "cd"
            random_state=self.random_state,
            # NOTE: do NOT pass normalize_components (not a valid kwarg)
            # avoid n_jobs for max compatibility unless you need it
        )
        spca.fit(Z)
        comps = spca.components_.astype(np.float64, copy=False)   # [K, N]

        # Optional but recommended: L2-normalize each component row
        # so that per-feature weights aren’t dominated by a single large-norm component.
        norms = np.linalg.norm(comps, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        comps = comps / norms

        return comps

    def _weights_from_components(self, comps: np.ndarray) -> np.ndarray:
        if self.weight_norm == "l1":
            w = np.sum(np.abs(comps), axis=0)
        else:
            w = np.sqrt(np.sum(comps ** 2, axis=0))
        if self.normalize_weights:
            s = float(np.sum(w))
            if s > 0:
                w = w / s
        return w

    def _analyze_and_report(self, step_ts: int):
        if self._X is None or self._filled < max(5, self.n_components + 2):
            return

        X = self._X[: self._filled]    # [R, N]
        names = self.feature_names or [f"f{i}" for i in range(X.shape[1])]

        comps = self._fit_spca(X)                      # [K, N]
        weights = self._weights_from_components(comps) # [N]
        order = np.argsort(-weights).tolist()

        # cache last snapshot for final save
        self._last_components = comps
        self._last_weights = weights
        self._last_ranked_idx = order

        # print summary
        if self.verbose:
            label = f"step={step_ts:,}"
            print(f"[FeatureSPCA] report @ {label}: reservoir_rows={self._filled:,} (of seen={self._rows_seen:,})  K={comps.shape[0]}  alpha={self.alpha}")
            k = min(self.print_top_k, len(order))
            print(f"  top-{k} features by SPCA weight:")
            for rank, idx in enumerate(order[:k], 1):
                print(f"    {rank:>2}. {names[idx]} (idx={idx})  w={weights[idx]:.6f}")

        # tensorboard summary (few scalars)
        if self.tensorboard:
            try:
                self.logger.record("spca/mean_weight", float(np.mean(weights)))
                self.logger.record("spca/max_weight", float(np.max(weights)))
                self.logger.record("spca/min_weight", float(np.min(weights)))
            except Exception:
                pass

        # periodic snapshot
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
                # ranked list
                pd.DataFrame({
                    "rank": np.arange(1, len(order)+1),
                    "idx": order,
                    "feature": [names[i] for i in order],
                    "weight": [weights[i] for i in order],
                }).to_csv(os.path.join(self.save_dir, f"spca_ranking_{stem}.csv"), index=False)
            except Exception:
                pass

    # ---------------- SB3 hooks ----------------

    def _on_rollout_end(self) -> bool:
        self.rollouts_seen += 1
        if self.rollouts_seen < self.warmup_rollouts:
            return True
        if (self.rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
            return True

        buf = self.model.rollout_buffer
        obs = self._to_np(buf.observations)  # [T, B, N] for (Recurrent)PPO
        if obs.ndim == 2:                    # rare fallback: [T, N] -> [T,1,N]
            obs = obs[:, None, :]
        T, B, N = obs.shape
        if self._N is None:
            self._N = N
            if self.feature_names is not None and len(self.feature_names) != N:
                raise ValueError(f"feature_names length {len(self.feature_names)} != N={N}")

        X_rows = obs.reshape(-1, N).astype(np.float32)

        # add to reservoir
        if self.reservoir_size == 0:
            self._X = X_rows.copy()
            self._filled = len(X_rows)
            self._rows_seen += len(X_rows)
        else:
            self._reservoir_add(X_rows)

        # print/log cadence aligned to rollout ends
        if self.print_every_steps is not None:
            if (self.num_timesteps - self._last_print_ts) >= self.print_every_steps:
                self._analyze_and_report(self.num_timesteps)
                self._last_print_ts = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        """
        Save a final snapshot of components, per-feature weights, and ranking.
        Uses the current reservoir; recomputes once to ensure the latest data are included.
        """
        if self._X is None or self._filled < max(5, self.n_components + 2):
            if self.verbose:
                print("[FeatureSPCA] training_end: not enough data to save final snapshot.")
            return

        # ensure we have the freshest snapshot
        self._analyze_and_report(self.num_timesteps)

        if not self.save_dir:
            if self.verbose:
                print("[FeatureSPCA] training_end: save_dir is None, skipping file save.")
            return

        names = self.feature_names or [f"f{i}" for i in range(self._X.shape[1])]
        comps = self._last_components
        weights = self._last_weights
        order = self._last_ranked_idx or list(np.argsort(-weights).tolist())

        stem = self.final_basename  # e.g., "final"
        # NPZ bundle
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
        # CSVs
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

        # after saving spca_final files...
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




# from sb3_contrib import RecurrentPPO


# cb_spca = FeatureSPCAFromRolloutCallback(
#     compute_every_rollouts=1,
#     warmup_rollouts=2,
#     print_every_steps=100_000,
#     print_top_k=20,
#     reservoir_size=20_000,
#     n_components=3,          # "top 3 SPCA components"
#     alpha=1.0,               # increase for sparser loadings
#     ridge_alpha=0.01,
#     max_iter=1000,
#     tol=1e-8,
#     method="lars",
#     weight_norm="l1",        # or "l2"
#     normalize_weights=True,
#     feature_names=[f"f{i}" for i in range(env.observation_space.shape[-1])],
#     save_dir="logs/spca_run1",
#     tensorboard=True,
#     verbose=1,
# )

# model.learn(total_timesteps=1_000_000, callback=cb_spca)
