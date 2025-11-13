from typing import Optional, List
import os
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

class FeatureCorrelationFromRolloutCallback(BaseCallback):
    """
    Diagnostyka korelacji dla (Recurrent)PPO z polityką MlpLstmPolicy.

    Na wybranych zakończeniach rolloutów (bez dodatkowych kroków środowiska):
      • Pobiera obserwacje rolloutu [T, B, N] oraz wektor celu (advantage/return/reward)
      • Dodaje wiersze do rezerwuaru o stałym rozmiarze (losowa próbka wszystkich dotąd widzianych wierszy)
      • W zadanych odstępach drukuje i loguje:
          - korelację Pearsona cecha→cel (uszeregowaną wg |r|)
          - korelacje cecha↔cecha (oznacza pary redundantne o |rho| >= próg)
      • Okresowo zapisuje zrzuty NPZ/CSV jeśli ustawiono save_dir
      • Zapisuje końcowy snapshot na końcu treningu (_on_training_end)

    Uwaga:
      - Rezerwuar ogranicza zużycie pamięci; statystyki uwagowe gdzie indziej pozostają bez zmian.
      - Odporne na kolumny o zerowej wariancji (dla nich korelacja = 0).
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
        """Konwertuj wejście (tensor/ndarray/lista) na ndarray NumPy."""
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
        """Dodaj wiersze do rezerwuaru metodą losowej próby (reservoir sampling)."""
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


import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

# -- helper: correlation pruning on reservoir rows --
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
    RecurrentPPO/PPO-friendly attention collector that aggregates over ALL training seen so far.

    Collects BOTH:
      - metric_importance (attention-only)
      - contrib_importance (contribution-aware)

    Cumulative episode-level stats per variant:
      • cum_mean_* [N] : Welford running mean of per-episode vectors
      • cum_var_*  [N] : Welford running variance (via M2)
      • freq_top_m_*[N]: count of top-m appearances per episode
      • ep_count       : total aggregated episodes

    You can choose which variant to use for printing/ranking/masking:
      - rank_source: "contrib" (default) | "metric"
      - mask_source: "contrib" (default) | "metric"
    """
    def __init__(
        self,
        compute_every_rollouts: int = 1,
        warmup_rollouts: int = 2,
        # printing
        print_every_steps=100_000,
        print_top_k: int = 15,
        top_m_for_frequency=None,  # default -> 2*sqrt(N) later
        feature_names=None,
        # feature pruning (optional)
        select_k=None,
        apply_mask: bool = False,
        corr_threshold: float = 0.95,
        # reservoir for correlation (keeps raw rows as seen by policy)
        reservoir_size: int = 20000,
        # which variant to use for printing/ranking/masking
        rank_source: str = "contrib",  # "contrib" | "metric"
        mask_source: str = "contrib",  # "contrib" | "metric"
        # saving/logging
        save_npz_path: str = "./logs/attn_cumulative/attn_cumulative_final.npz",
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

        # variant selection
        self.rank_source = str(rank_source).lower()
        self.mask_source = str(mask_source).lower()
        assert self.rank_source in {"contrib", "metric"}
        assert self.mask_source in {"contrib", "metric"}

        # cumulative stats (shared episode counter)
        self.ep_count = 0
        # metric-only (attention weights)
        self.cum_mean_metric = None
        self.cum_M2_metric = None
        self.freq_top_m_metric = None
        # contribution-aware
        self.cum_mean_contrib = None
        self.cum_M2_contrib = None
        self.freq_top_m_contrib = None

        # printing cadence
        self._last_print_ts = 0

        # reservoir for correlation
        self.reservoir_size = int(reservoir_size)
        self._reservoir = None
        self._reservoir_filled = 0
        self._rows_seen = 0

        self.rollouts_seen = 0
        self.save_npz_path = save_npz_path
        self._cum_snapshots = []  # list of dicts

    # -------- utils --------
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

    # -------- SB3 hooks --------
    def _on_rollout_end_bp(self) -> bool:
        """
        On each rollout, always add observations to the reservoir.
        Then, after warmup + cadence, optionally run SPCA analysis.
        """
        self.rollouts_seen += 1

        # ---- 1) grab rollout observations and update reservoir ----
        buf = self.model.rollout_buffer
        obs = self._to_np(buf.observations)  # PPO / RecurrentPPO: [T, B, N]
        if obs.ndim == 2:                     # rare fallback: [T, N] -> [T,1,N]
            obs = obs[:, None, :]
        T, B, N = obs.shape

        if self._N is None:
            self._N = N
            if self.feature_names is not None and len(self.feature_names) != N:
                raise ValueError(
                    f"feature_names length {len(self.feature_names)} != N={N}"
                )

        X_rows = obs.reshape(-1, N).astype(np.float32)  # [T*B, N]

        # Always add to reservoir
        if self.reservoir_size == 0:
            # special case: keep all rows
            self._X = X_rows.copy()
            self._filled = len(X_rows)
            self._rows_seen += len(X_rows)
        else:
            self._reservoir_add(X_rows)

        # ---- 2) warmup / cadence for SPCA computation only ----
        if self.rollouts_seen < self.warmup_rollouts:
            return True
        if (self.rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
            return True

        # ---- 3) optionally analyze + save intermediate snapshot ----
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
        if obs.ndim == 2:  # fallback if buffer squeezed
            obs = obs[:, None, :]
            ep_starts = ep_starts[:, None]

        T, B, N = obs.shape
        device = self.model.policy.device
        fe = self.model.policy.features_extractor

        # forward once to populate attention diagnostics
        was_training = fe.training
        fe.eval()
        with th.no_grad():
            x = th.as_tensor(obs, device=device, dtype=th.float32)  # [T,B,N]
            _ = fe(x)  # runs extractor; fills .metric_importance / .contrib_importance
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

        # collect per-episode means for each variant
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

        # ---- update cumulative Welford stats ----
        # initialize arrays lazily from first available variant
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

        # episode counter increments by number of episodes in this rollout (consistent for both)
        # use whichever per-ep array is not None to count episodes
        E_r = 0
        if per_ep_metric is not None:
            E_r = per_ep_metric.shape[0]
        elif per_ep_contrib is not None:
            E_r = per_ep_contrib.shape[0]
        self.ep_count += int(E_r)

        # vectorized Welford update for each episode
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

        # top-m frequency updates
        m = int(self.top_m_for_frequency or max(1, min(N, 2 * int(np.ceil(np.sqrt(N))))))
        if per_ep_metric is not None:
            topm_idx = np.argsort(-per_ep_metric, axis=1)[:, :m]
            for row in topm_idx:
                self.freq_top_m_metric[row] += 1.0
        if per_ep_contrib is not None:
            topm_idx = np.argsort(-per_ep_contrib, axis=1)[:, :m]
            for row in topm_idx:
                self.freq_top_m_contrib[row] += 1.0

        # keep reservoir for correlation pruning (rows as seen by policy)
        self._reservoir_add(obs.reshape(-1, N))

        # ---- logging to TensorBoard ----
        # metric
        for i, val in enumerate(self.cum_mean_metric):
            self.logger.record(f"attn/mean_f{i}/M", float(val))
        self.logger.record("attn/episodes/M", int(self.ep_count))
        self.logger.record("attn/mean_all/M", float(self.cum_mean_metric.mean()))

        # contrib
        for i, val in enumerate(self.cum_mean_contrib):
            self.logger.record(f"attn/mean_f{i}/C", float(val))
        self.logger.record("attn/episodes/C", int(self.ep_count))
        self.logger.record("attn/mean_all/C", float(self.cum_mean_contrib.mean()))


        # ---- optional: apply mask using cumulative ranking + reservoir correlation ----
        if (self.select_k is not None) and self.apply_mask:
            # choose source for mask
            if self.mask_source == "contrib" and self.cum_mean_contrib is not None:
                freq = self.freq_top_m_contrib / max(1, self.ep_count)
                var  = (self.cum_M2_contrib / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_contrib)
                base = self.cum_mean_contrib
            else:
                freq = self.freq_top_m_metric / max(1, self.ep_count)
                var  = (self.cum_M2_metric / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_metric)
                base = self.cum_mean_metric

            order = np.lexsort((var, -base, -freq))   # best -> worst
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

        # ---- cumulative PRINT every n steps (aligned to rollout end) ----
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

        # snapshot for saving
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
        # consolidate snapshots (pad missing keys with zeros for consistent saving)
        steps = np.array([s["step"] for s in self._cum_snapshots], dtype=np.int64)

        def _stack_or_zeros(key, N):
            arrs = []
            for s in self._cum_snapshots:
                if key in s:
                    arrs.append(s[key])
                else:
                    arrs.append(np.zeros(N, dtype=np.float32))
            return np.stack(arrs, axis=0)

        # infer N from whichever mean exists
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
            # metric-only series
            means_metric=means_metric, vars_metric=vars_metric, freqs_metric=freqs_metric,
            # contribution-aware series
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