# spca_reweight.py
import numpy as np
import torch as th
import torch.nn as nn
from typing import Optional
from gym import spaces
from gym.spaces.utils import flatdim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
try:
    # SB3 >= 1.5: lets you push matplotlib figures to TensorBoard
    from stable_baselines3.common.logger import Figure
    _HAS_SB3_FIGURE = True
except Exception:
    _HAS_SB3_FIGURE = False

# HERE I HAVE THE CODE for spca rewighting while training - online
class SPCAReweightingExtractor(BaseFeaturesExtractor):
    """
    Elementwise reweighting of input features using weights derived from SPCA/PCA.
    Output dim == input dim (no shape change).

    y = ((x - mu) / sigma) * w      if standardize=True
    y = x * w                        if standardize=False (assume upstream normalization)
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        standardize: bool = True,       # set False if you already use VecNormalize(obs=True)
        min_sigma: float = 1e-8,
    ):
        in_dim = flatdim(observation_space)
        super().__init__(observation_space, features_dim=in_dim)
        self.in_dim = in_dim
        self.standardize = bool(standardize)
        self.min_sigma = float(min_sigma)

        # Buffers so we can hot-swap from the callback
        self.register_buffer("mu", th.zeros(in_dim))
        self.register_buffer("sigma", th.ones(in_dim))
        self.register_buffer("w", th.ones(in_dim))   # per-feature weights
        self.enabled = False

    def enable(self, flag: bool = True):
        self.enabled = bool(flag)

    @th.no_grad()
    def update_reweighting(self, w: np.ndarray, mu: Optional[np.ndarray], sigma: Optional[np.ndarray], enable: bool = True):
        assert w.shape == (self.in_dim,)
        device = self.mu.device
        self.w.copy_(th.as_tensor(w, dtype=th.float32, device=device))
        if self.standardize:
            assert mu is not None and sigma is not None
            self.mu.copy_(th.as_tensor(mu, dtype=th.float32, device=device))
            self.sigma.copy_(th.as_tensor(np.maximum(sigma, self.min_sigma), dtype=th.float32, device=device))
        self.enable(enable)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations
        if not self.enabled:
            # Pass-through until weights are set, keeping the same dimensionality.
            return x
        if self.standardize:
            x = (x - self.mu) / self.sigma
        return x * self.w


# class SPCAReweightCallback(BaseCallback):
#     """
#     After a warmup, fit SPCA (fallback PCA) on recent observations and compute
#     per-feature importances w_j. Normalize w to mean=1 and (optionally) smooth
#     with EMA to avoid sudden jumps. Push to SPCAReweightingExtractor.

#     Importance:
#       - PCA:   imp_j = Σ_k EVR_k * (V[k,j]**2)
#       - SPCA:  imp_j = Σ_k Var(Z_k) * (C[k,j]**2)
#         where C = components, Z = transform(Xs)

#     Shapes:
#       X ∈ ℝ^{N×D}, D=7
#     """
#     def __init__(
#         self,
#         warmup_steps: int,          # wait before first fit (e.g., your PPO n_steps)
#         hist_steps: int,            # fit on the last K samples
#         n_components: int = 4,
#         update_every: int = 0,      # 0 = fit once; else re-fit every N env steps
#         prefer_spca: bool = True,
#         ema_beta: float = 0.5,      # 0=no smoothing, 0.5=moderate, 0.9=slow
#         clip_weights: tuple = (0.25, 4.0),  # clip final w to this range
#         assume_normalized_input: bool = False,  # True if using VecNormalize(obs=True)
#         orig_metric_dim: int = 7,   # if obs has stacked history (k*7), we still fit on the last 7
#         verbose: int = 0,
#     ):
#         super().__init__(verbose=verbose)
#         self.warmup_steps = int(warmup_steps)
#         self.hist_steps = int(hist_steps)
#         self.n_components = int(n_components)
#         self.update_every = int(update_every)
#         self.prefer_spca = bool(prefer_spca)
#         self.ema_beta = float(ema_beta)
#         self.clip_weights = clip_weights
#         self.assume_normalized_input = bool(assume_normalized_input)
#         self.orig_metric_dim = int(orig_metric_dim)
#         self._last_fit_at = None

#     def _on_training_start(self) -> None:
#         fe = getattr(self.model.policy, "features_extractor", None)
#         if not isinstance(fe, SPCAReweightingExtractor):
#             raise RuntimeError("Policy must use SPCAReweightingExtractor for SPCAReweightCallback.")
#         if self.verbose:
#             print("[SPCA-RW] Ready. standardize=", fe.standardize)

#     def _flatten_buffer_obs(self) -> np.ndarray:
#         """
#         Pull observations from the rollout buffer and return (N, D) float32.
#         Works across SB3 versions where observations can be np.ndarray or torch.Tensor.
#         """
#         buf = self.model.rollout_buffer
#         obs = buf.observations  # could be np.ndarray, torch.Tensor, or dict for Dict spaces

#         # Handle Dict observation spaces (not your case, but robust)
#         if isinstance(obs, dict):
#             # choose a primary key or concat; here we pick the first key deterministically
#             first_key = sorted(obs.keys())[0]
#             obs = obs[first_key]

#         # Convert to numpy without assuming PyTorch
#         try:
#             import torch as th  # optional (will succeed if torch is installed)
#             if isinstance(obs, th.Tensor):
#                 X = obs.detach().cpu().numpy()
#             else:
#                 X = np.asarray(obs)
#         except Exception:
#             X = np.asarray(obs)

#         # Ensure shape is (n_steps, n_envs, obs_dim_flat)
#         if X.ndim >= 3:
#             X = X.reshape(X.shape[0], X.shape[1], -1)
#         else:
#             # Some older SB3 store as (n_steps * n_envs, obs_dim) already
#             X = X.reshape(-1, X.shape[-1])

#         # Flatten time/env dims -> (N, D)
#         X = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)

#         # If your obs is a flattened history (k*7), and you only want the most recent 7
#         if X.shape[1] % self.orig_metric_dim == 0 and X.shape[1] != self.orig_metric_dim:
#             start = X.shape[1] - self.orig_metric_dim
#             X = X[:, start:]

#         return X


#     def _fit_weights(self, X: np.ndarray):
#         D = X.shape[1]
#         # Standardize for stable SPCA/PCA unless upstream already did it
#         if self.assume_normalized_input:
#             mu = np.zeros(D, dtype=np.float32)
#             sigma = np.ones(D, dtype=np.float32)
#             Xs = X.astype(np.float32, copy=False)
#         else:
#             mu = X.mean(axis=0)
#             sigma = X.std(axis=0)
#             sigma = np.where(sigma < 1e-8, 1.0, sigma)
#             Xs = (X - mu) / sigma

#         used_spca = False
#         imp = None

#         if self.prefer_spca:
#             try:
#                 from sklearn.decomposition import MiniBatchSparsePCA as SparsePCA
#                 spca = SparsePCA(n_components=self.n_components, alpha=1.0, batch_size=256, random_state=0)
#                 spca.fit(Xs)
#                 Z = spca.transform(Xs)                    # (N, M)
#                 varZ = np.var(Z, axis=0) + 1e-12          # component energies
#                 C = spca.components_                      # (M, D)
#                 imp = (C**2).T @ varZ                     # (D,)
#                 used_spca = True
#             except Exception as e:
#                 if self.verbose:
#                     print(f"[SPCA-RW] SPCA failed ({e}); falling back to PCA.")

#         if imp is None:
#             # PCA via SVD to get EVR
#             U, S, Vt = np.linalg.svd(Xs, full_matrices=False)  # Xs ≈ U S V^T
#             # explained variance per component = S^2 / (N-1)
#             N = max(2, Xs.shape[0])
#             ev = (S**2) / (N - 1)
#             evr = ev / (ev.sum() + 1e-12)
#             M = min(self.n_components, Vt.shape[0])
#             V = Vt[:M, :]                                   # (M, D)
#             w_comp = evr[:M]                                # (M,)
#             imp = (V**2).T @ w_comp                         # (D,)

#         # Normalize: mean(w)=1, clip extremes
#         imp = np.maximum(imp, 1e-12)
#         w = imp / np.mean(imp)
#         w = np.clip(w, self.clip_weights[0], self.clip_weights[1])

#         return w.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32), used_spca

#     def _maybe_update(self):
#         total = self.num_timesteps
#         if total < self.warmup_steps:
#             return
#         if self._last_fit_at is not None and self.update_every > 0:
#             if total - self._last_fit_at < self.update_every:
#                 return

#         X = self._flatten_buffer_obs()
#         if X.shape[0] < max(8, self.hist_steps):
#             # not enough data yet
#             return

#         X = X[-self.hist_steps:, :]
#         w_new, mu, sigma, used_spca = self._fit_weights(X)

#         fe: SPCAReweightingExtractor = self.model.policy.features_extractor
#         if fe.enabled and self.ema_beta > 0.0:
#             # smooth with EMA to avoid sudden jumps
#             w_old = fe.w.detach().cpu().numpy()
#             w_new = self.ema_beta * w_new + (1 - self.ema_beta) * w_old

#         fe.update_reweighting(w=w_new, mu=mu, sigma=sigma, enable=True)
#         self._last_fit_at = self.num_timesteps

#         if self.verbose:
#             kind = "SPCA" if used_spca else "PCA"
#             print(f"[SPCA-RW] Activated {kind} reweighting at step {self._last_fit_at}. w={np.round(w_new,3)}")

#     def _on_rollout_end(self) -> None:
#         self._maybe_update()

#     def _on_step(self) -> bool:
#         # satisfy abstract method; optionally allow mid-rollout refresh
#         if self.update_every > 0:
#             self._maybe_update()
#         return True

import os
import matplotlib.pyplot as plt
# -----------------------------------------------
# SPCAReweightFullHistoryCallback (full version)
# -----------------------------------------------
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback

try:
    from stable_baselines3.common.logger import Figure
    _HAS_SB3_FIGURE = True
except Exception:
    _HAS_SB3_FIGURE = False


class SPCAReweightCallback(BaseCallback):
    """
    Accumulates all observations seen so far (optionally capped), fits SPCA/PCA
    on the entire history at the chosen cadence, converts component energies to
    per-feature weights w (mean=1, clipped), pushes them to the
    SPCAReweightingExtractor, and visualizes.

    Use when n_steps is large (e.g., 2048) and you want PCA based on everything
    observed up to the current time j_step, then re-fit again on the expanded history.
    """

    def __init__(
        self,
        warmup_steps: int = 0,           # minimum total samples before first fit (across all rollouts)
        min_fit_samples: int = 256,      # also require at least this many samples to fit
        refit_every_rollouts: int = 10_000,   # re-fit after this many completed rollouts
        max_history: int = None,         # cap history to last N samples (None = full history)
        n_components: int = 4,
        prefer_spca: bool = True,
        ema_beta: float = 0.5,           # smooth weights: w <- beta*w_new + (1-beta)*w_old
        clip_weights: tuple = (0.25, 4.0),
        assume_normalized_input: bool = False,  # True if you use VecNormalize(obs=True)
        orig_metric_dim: int = 7,        # if obs is stacked (k*7), fit only on the last 7 per sample
        verbose: int = 1,
        # Visualization:
        viz_dir: str = None,             # save PNGs here (optional)
        feature_names: list = None,      # labels for bars
    ):
        super().__init__(verbose=verbose)
        self.warmup_steps = int(warmup_steps)
        self.min_fit_samples = int(min_fit_samples)
        self.refit_every_rollouts = int(refit_every_rollouts)
        self.max_history = None if max_history is None else int(max_history)
        self.n_components = int(n_components)
        self.prefer_spca = bool(prefer_spca)
        self.ema_beta = float(ema_beta)
        self.clip_weights = clip_weights
        self.assume_normalized_input = bool(assume_normalized_input)
        self.orig_metric_dim = int(orig_metric_dim)

        # Accumulators
        self._history = None            # np.ndarray of shape (N, D)
        self._samples_seen = 0
        self._rollouts_since_fit = 0
        self._last_w = None

        # Viz
        self.viz_dir = viz_dir
        self.feature_names = feature_names
        if self.viz_dir:
            os.makedirs(self.viz_dir, exist_ok=True)

    # ---------- SB3 hooks ----------
    def _on_training_start(self) -> None:
        if self.verbose:
            print("[SPCA-RW/Full] ready. TB Figure:", _HAS_SB3_FIGURE,
                  "| warmup_samples:", self.warmup_steps,
                  "| min_fit_samples:", self.min_fit_samples,
                  "| refit_every_rollouts:", self.refit_every_rollouts)

    def _on_rollout_end(self) -> None:
        # 1) append current rollout obs to global history
        X = self._extract_buffer_obs()
        if X is None or X.size == 0:
            return
        self._append_history(X)

        # 2) decide whether to fit now
        self._rollouts_since_fit += 1
        if (self._samples_seen >= max(self.warmup_steps, self.min_fit_samples)
                and self._rollouts_since_fit >= self.refit_every_rollouts):
            self._fit_and_apply()  # this will also visualize
            self._rollouts_since_fit = 0

    def _on_step(self) -> bool:
        # We do fitting at rollout end; keep this hook as a no-op to satisfy BaseCallback
        return True

    # ---------- Accumulation ----------
    def _extract_buffer_obs(self) -> np.ndarray:
        """Get obs from rollout buffer as (N, D) float32; reduce to last 7 if stacked."""
        buf = self.model.rollout_buffer
        obs = buf.observations
        # Dict spaces: pick first key
        if isinstance(obs, dict):
            key = sorted(obs.keys())[0]
            obs = obs[key]

        # Convert to numpy
        X = None
        try:
            import torch as th
            if isinstance(obs, th.Tensor):
                X = obs.detach().cpu().numpy()
        except Exception:
            X = None
        if X is None:
            X = np.asarray(obs)

        # (n_steps, n_envs, D) -> (N, D)
        if X.ndim >= 3:
            X = X.reshape(X.shape[0], X.shape[1], -1)
        X = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)

        # If obs is stacked (k*7), keep only last 7 for fitting
        if self.orig_metric_dim > 0 and X.shape[1] % self.orig_metric_dim == 0 and X.shape[1] != self.orig_metric_dim:
            start = X.shape[1] - self.orig_metric_dim
            X = X[:, start:]

        return X

    def _append_history(self, X: np.ndarray) -> None:
        """Append new samples to the global history (optionally capped)."""
        if self._history is None:
            self._history = X.copy()
        else:
            self._history = np.concatenate([self._history, X], axis=0)

        # Cap history length if requested
        if self.max_history is not None and self._history.shape[0] > self.max_history:
            self._history = self._history[-self.max_history:, :]

        self._samples_seen = int(self._history.shape[0])

    # ---------- Fitting & application ----------
    def _fit_and_apply(self) -> None:
        Xhist = self._history
        if Xhist is None or Xhist.shape[0] < max(8, self.min_fit_samples):
            return

        w_new, mu, sigma, used_spca = self._fit_weights(Xhist)

        # EMA smoothing
        if self._last_w is not None and self.ema_beta > 0.0:
            w_new = self.ema_beta * w_new + (1.0 - self.ema_beta) * self._last_w

        # Push to extractor (must expose update_reweighting)
        fe = self.model.policy.features_extractor
        fe.update_reweighting(w=w_new, mu=mu, sigma=sigma, enable=True)

        self._last_w = w_new.copy()

        if self.verbose:
            kind = "SPCA" if used_spca else "PCA"
            print(f"[SPCA-RW/Full] fitted {kind} on N={Xhist.shape[0]} samples.")

        self._viz_weights(w_new, "SPCA/PCA (full-history)", self._samples_seen)

    def _fit_weights(self, X: np.ndarray):
        """Fit SPCA/PCA on X and compute per-feature weights; return (w, mu, sigma, used_spca)."""
        D = X.shape[1]
        if self.assume_normalized_input:
            mu = np.zeros(D, dtype=np.float32)
            sigma = np.ones(D, dtype=np.float32)
            Xs = X.astype(np.float32, copy=False)
        else:
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            Xs = (X - mu) / sigma

        used_spca = False
        imp = None

        if self.prefer_spca:
            try:
                from sklearn.decomposition import MiniBatchSparsePCA as SparsePCA
                spca = SparsePCA(n_components=self.n_components, alpha=1.0, batch_size=256, random_state=0)
                spca.fit(Xs)
                Z = spca.transform(Xs)
                varZ = np.var(Z, axis=0) + 1e-12
                C = spca.components_
                imp = (C ** 2).T @ varZ
                used_spca = True
            except Exception as e:
                if self.verbose:
                    print(f"[SPCA-RW/Full] SPCA failed ({e}); falling back to PCA.")

        if imp is None:
            U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
            N = max(2, Xs.shape[0])
            ev = (S ** 2) / (N - 1)
            evr = ev / (ev.sum() + 1e-12)
            M = min(self.n_components, Vt.shape[0])
            V = Vt[:M, :]
            w_comp = evr[:M]
            imp = (V ** 2).T @ w_comp

        imp = np.maximum(imp, 1e-12)
        w = imp / np.mean(imp)
        w = np.clip(w, self.clip_weights[0], self.clip_weights[1])
        return w.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32), used_spca

    # ---------- Visualization ----------
    def _viz_weights(self, w: np.ndarray, kind: str, step: int):
        names = self.feature_names or [f"f{i}" for i in range(len(w))]
        x = np.arange(len(w))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        try:
            bars = ax.bar(x, w)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_ylim(0, max(4.0, float(w.max()) * 1.15))
            ax.set_ylabel("weight")
            ax.set_title(f"{kind} @ sample {step} (mean=1)")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            fig.subplots_adjust(bottom=0.3)
            # 🔑 Add value labels on top of bars
            for rect in bars:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height,
                    f"{height:.2f}",   # format with 2 decimals
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

            if _HAS_SB3_FIGURE:
                self.logger.record("spca/weights_bar", Figure(fig, close=False),
                                   exclude=("stdout", "log", "json"))
            for j, val in enumerate(w):
                self.logger.record(f"spca/w_{j}", float(val))

            if self.viz_dir:
                out = os.path.join(self.viz_dir, f"spca_weights_samples_{step}.png")
                fig.savefig(out, bbox_inches="tight")
        finally:
            plt.close(fig)

def save_model(model, env):
        # 1) Model (policy + extractor buffers)
    model.save("ppo_spca")

    # 2) VecNormalize stats (if you used VecNormalize)
    #    'venv' is the underlying env returned by make_vec_env before VecNormalize,
    #    but here we assume `env` is your training env with VecNormalize on top.
    env.save("vecnorm.pkl")

    # 3) Optional: SPCA snapshot (nice for plots/debug)
    fe = model.policy.features_extractor
    np.savez_compressed(
        "spca_snapshot.npz",
        w=fe.w.detach().cpu().numpy(),
        mu=fe.mu.detach().cpu().numpy(),
        sigma=fe.sigma.detach().cpu().numpy(),
        standardize=np.array([fe.standardize], dtype=np.bool_),  # metadata
        in_dim=np.array([fe.in_dim], dtype=np.int32),
    )

def load_model(args, extra_args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from run import build_env
    # 1) Rebuild the env exactly like training (same wrappers/order)
    eval_env = build_env(args, extra_args)  # your factory

    # 2) Restore VecNormalize stats and freeze them
    #TODO commented it out cause they cause rewards to be higher
    # eval_env = VecNormalize.load("vecnorm.pkl", eval_env)
    # eval_env.training = False
    # eval_env.norm_reward = False  # get true env rewards during eval

    # 3) Load the model; make sure the extractor class is importable
    model = PPO.load(
        "ppo_spca",
        custom_objects={"features_extractor_class": SPCAReweightingExtractor},
        device="cpu",  # or "cuda"
    )

    # 4) IMPORTANT: ensure the extractor is enabled (bools aren’t saved by state_dict)
    fe = model.policy.features_extractor
    if hasattr(fe, "enable"):
        fe.enable(True)

    # 5) (Optional) Push an explicit snapshot if you want to be extra sure / inspect
    snap = np.load("spca_snapshot.npz")
    fe.update_reweighting(snap["w"], snap["mu"], snap["sigma"], enable=True)
    return model, eval_env



def train_model(env, args):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO   # if using LSTM
    # If you use VecNormalize(obs=True), set standardize=False in the extractor
    use_vecnorm = True

    policy_kwargs = dict(
        features_extractor_class=SPCAReweightingExtractor,
        features_extractor_kwargs=dict(
            standardize=not use_vecnorm
        ), 
        lstm_hidden_size=64, #decrease hidden size further to show that less metrics is better
    #shared_lstm=True,
    net_arch=[dict(pi=[16, 16], vf=[16, 16])],
    activation_fn=nn.ReLU,
    ortho_init=True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",                 # or "MlpLstmPolicy" with RecurrentPPO
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=0.00003,
        n_steps=256,
        batch_size=256,
        #n_steps=2048,
        #0.00003, # 0.00003 #for LSTM model I changed lr to 0.0003, #explained variance more stable for LSTM when 0.00003 than 0.0003, but still grows in the end
        #LSTM started learning with 0.0003 after 250k steps, incredibly slow

        vf_coef=1,
        clip_range_vf=10.0,
        #clip_range=0.2,
        max_grad_norm=1,
        gamma=0.95,
        ent_coef=0.001,
        clip_range=0.05,
        verbose=1,
        seed=int(args.seed),
        tensorboard_log='./output_malota')
    

    

    #model.learn(total_timesteps=args.num_timesteps/2, callback=spca_cb)
    return model
