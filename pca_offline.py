# pca_tooling.py
import os
from typing import List, Tuple, Optional, Sequence, Union
from utils import FEATURE_NAMES

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gym
from gym import spaces

# ---------- 1) Rollout + collection ----------

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch as th
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _extract_obs_array(obs) -> np.ndarray:
    """Return obs as (n_envs, obs_dim) numpy array."""
    if isinstance(obs, dict):
        # Pick a key deterministically (customize for Dict spaces)
        key = sorted(obs.keys())[0]
        obs = obs[key]
    arr = _to_numpy(obs)
    if arr.ndim == 1:
        arr = arr[None, ...]
    return arr

from typing import List
import numpy as np

def _to_python(obj):
    """Convert NumPy types/arrays to plain Python (int/float/list)."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _sample_python_action(space):
    """space.sample() → Python-native action (no NumPy scalars)."""
    return _to_python(space.sample())

def collect_observations_backup(
    model,                      # kept for signature compatibility (unused)
    env,
    n_steps: int = 10_000,
    deterministic: bool = True, # unused
    progress: bool = True,
) -> np.ndarray:
    """
    Roll the env for n_steps with RANDOM actions and collect observations.
    Works with SB3 VecEnv and classic gym.Env. Returns X with shape (N, D).
    """
    obs = env.reset()
    X_list: List[np.ndarray] = []

    for t in range(n_steps):
        obs_arr = _extract_obs_array(obs)  # (n_envs, D)
        X_list.append(obs_arr)

        # --- RANDOM actions, but Python-native (no numpy.int32) ---
        if hasattr(env, "num_envs"):  # VecEnv
            action = [_sample_python_action(env.action_space) for _ in range(env.num_envs)]
        else:                         # classic gym.Env
            action = _sample_python_action(env.action_space)

        obs, reward, done, info = env.step(action)

        # Handle classic gym (bool) vs VecEnv (array/list of bools)
        if isinstance(done, (list, tuple, np.ndarray)):
            if np.any(done):
                obs = env.reset()
        else:
            if done:
                obs = env.reset()

        if progress and (t + 1) % 1000 == 0:
            print(f"[collect] {t+1}/{n_steps} steps")

    X = np.concatenate(X_list, axis=0)  # (N, D)
    return X.astype(np.float32, copy=False)



def collect_observations(
    model,
    env,
    n_steps: int = 10_000,
    deterministic: bool = True,
    progress: bool = True,
) -> np.ndarray:
    """
    Roll the given policy in `env` for n_steps and collect observations.
    Works with SB3 policies and both VecEnv and classic gym.Env.
    Returns X with shape (N, D).
    """
    # Reset env (VecEnv returns batch of obs)
    obs = env.reset()
    X_list: List[np.ndarray] = []

    for t in range(n_steps):
        obs_arr = _extract_obs_array(obs)  # (n_envs, D)
        X_list.append(obs_arr)

        action, _state = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        # Handle classic gym (bool) vs VecEnv (array of bools)
        if isinstance(done, (list, tuple, np.ndarray)):
            if np.any(done):
                obs = env.reset()
        else:
            if done:
                obs = env.reset()

        if progress and (t + 1) % 1000 == 0:
            print(f"[collect] {t+1}/{n_steps} steps")

    X = np.concatenate(X_list, axis=0)  # (N, D)
    return X.astype(np.float32, copy=False)

# ---------- 2) PCA analysis + plots ----------

def pca_analysis_backup(
    X: np.ndarray,
    n_components: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
    show: bool = True,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Run PCA on X (N,D) and produce:
      - scree plot (explained variance ratio per component)
      - PC1 vs PC2 scatter
      - per-feature importance bar chart
    Returns a dict with PCA results and importance.
    """
    N, D = X.shape
    if n_components is None:
        n_components = min(D, 10)

    # Standardization (mean/var) is built into PCA on centered data;
    # here we do mean-centering; variance scaling is not applied
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)  # guard tiny std
    Xc = (X - mu) / sigma

    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    Z = pca.fit_transform(Xc)             # (N, M)
    evr = pca.explained_variance_ratio_   # (M,)
    comps = pca.components_               # (M, D) rows are PCs

    # Per-feature importance: sum over components of squared loadings weighted by EVR
    # imp_j = sum_k (evr_k * comps[k,j]^2)
    importance = (comps ** 2).T @ evr     # (D,)
    # Normalize to mean=1 for interpretability
    importance = importance / (importance.mean() + 1e-12)

    # --- Plots ---
    names = list(feature_names) if FEATURE_NAMES is not None else [f"f{i}" for i in range(D)]
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    # 2.1 Scree plot
    fig1, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax1.plot(np.arange(1, len(evr) + 1), evr, marker="o")
    ax1.set_xlabel("Składowa główna")
    ax1.set_ylabel("Udział wyjaśnionej wariancji")
    ax1.set_title('Wykres "scree plot"')
    ax1.grid(True, linestyle="--", alpha=0.4)
    if save_dir:
        fig1.savefig(os.path.join(save_dir, "pca_scree.png"), dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig1)

    # 2.2 PC1 vs PC2 scatter (if at least 2 comps)
    if Z.shape[1] >= 2:
        fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax2.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_title("PC1 vs PC2")
        ax2.grid(True, linestyle="--", alpha=0.4)
        if save_dir:
            fig2.savefig(os.path.join(save_dir, "pca_scatter_pc1_pc2.png"), dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig2)

    # 2.3 Per-feature importance bar
    fig3, ax3 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    x = np.arange(D)
    bars = ax3.bar(x, importance)                     # ← keep handle
    for rect in bars:
        h = rect.get_height()
        ax3.annotate(f"{h:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2 if h >= 0 else -2),  # small offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if h >= 0 else "top")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=35, ha="right")
    ax3.set_ylim(0.9, 1.2)
    ax3.set_ylabel("Względna istotność cech")
    ax3.set_title("Istotność cech na podstawie PCA")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
    if save_dir:
        fig3.savefig(os.path.join(save_dir, "pca_feature_importance.png"), dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig3)

    return dict(
        pca=pca,
        mu=mu,
        sigma=sigma,
        scores=Z,
        evr=evr,
        components=comps,
        importance=importance,
        feature_names=names,
    )
def pca_analysis(
    X: np.ndarray,
    n_components: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
    show: bool = True,
    save_dir: Optional[str] = None,
    topk: int = 5,                 # ← NEW: how many top PCs to visualize
    weight_by_evr: bool = True,    # ← NEW: weight contributions by EVR
) -> dict:
    """
    Run PCA on X (N,D) and produce:
      - scree plot (explained variance ratio per component)
      - PC1 vs PC2 scatter
      - per-feature importance bar chart (all PCs, mean-normalized)
      - NEW: stacked bars with feature contributions to top-k PCs
    Returns a dict with PCA results and importances.
    """
    N, D = X.shape
    if n_components is None:
        n_components = min(D, 10)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xc = (X - mu) / sigma

    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    Z = pca.fit_transform(Xc)             # (N, M)
    evr = pca.explained_variance_ratio_   # (M,)
    comps = pca.components_               # (M, D) rows are PCs

    # Original aggregate importance across ALL PCs (EVR-weighted, mean=1)
    importance_all = (comps ** 2).T @ evr      # (D,)
    importance = importance_all / (importance_all.mean() + 1e-12)

    # Names (fixing the bug: use function arg, not FEATURE_NAMES)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(D)]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 1) Scree
    fig1, ax1 = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax1.plot(np.arange(1, len(evr) + 1), evr, marker="o")
    ax1.set_xlabel("Składowa główna")
    ax1.set_ylabel("Udział wyjaśnionej wariancji")
    ax1.set_title('Wykres "scree plot"')
    ax1.grid(True, linestyle="--", alpha=0.4)
    if save_dir:
        fig1.savefig(os.path.join(save_dir, "pca_scree.png"), dpi=120, bbox_inches="tight")
    plt.show() if show else plt.close(fig1)

    # 2) PC1 vs PC2
    if Z.shape[1] >= 2:
        fig2, ax2 = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax2.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7)
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
        ax2.set_title("PC1 vs PC2")
        ax2.grid(True, linestyle="--", alpha=0.4)
        if save_dir:
            fig2.savefig(os.path.join(save_dir, "pca_scatter_pc1_pc2.png"), dpi=120, bbox_inches="tight")
        plt.show() if show else plt.close(fig2)

    # 3) Per-feature importance (ALL PCs, mean-normalized) — original plot
    fig3, ax3 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    x = np.arange(D)
    bars = ax3.bar(x, importance)
    # labels (fallback for older Matplotlib)
    for rect in bars:
        h = rect.get_height()
        ax3.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 2 if h >= 0 else -2), textcoords="offset points",
                     ha="center", va="bottom" if h >= 0 else "top")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=35, ha="right")
    ax3.set_ylabel("Względna istotność cech (wszystkie PC)")
    ax3.set_title("Istotność cech na podstawie PCA (mean=1)")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
    if save_dir:
        fig3.savefig(os.path.join(save_dir, "pca_feature_importance_all.png"), dpi=120, bbox_inches="tight")
    plt.show() if show else plt.close(fig3)

    # 4) NEW: Contribution of features to TOP-k PCs (stacked bars)
    k = int(max(1, min(topk, comps.shape[0])))
    # per-PC-per-feature contribution
    if weight_by_evr:
        contrib_k = (comps[:k, :] ** 2) * evr[:k, None]   # shape (k, D)
        y_label = "Kontrybucja do top-k PC (ważona EVR)"
        title = f"Kontrybucja cech do top-{k} składowych głównych"
    else:
        contrib_k = (comps[:k, :] ** 2)
        y_label = "Kontrybucja do top-k PC (|ładunki|^2)"
        title = f"Kontrybucja cech do top-{k} składowych głównych"

    totals = contrib_k.sum(axis=0)  # (D,)

    fig4, ax4 = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    bottom = np.zeros(D, dtype=float)
    for i in range(k):
        ax4.bar(x, contrib_k[i], bottom=bottom, label=f"PC{i+1}")
        bottom += contrib_k[i]

    # labels on top of stacks
    for xi, h in zip(x, totals):
        ax4.annotate(f"{h:.2e}" if h < 0.01 else f"{h:.2f}",
                     xy=(xi, h), xytext=(0, 2), textcoords="offset points",
                     ha="center", va="bottom")

    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=35, ha="right")
    ax4.set_ylabel(y_label)
    ax4.set_title(title)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax4.legend(ncol=min(4, k), fontsize=9)
    ax4.margins(y=0.05)

    if save_dir:
        fig4.savefig(os.path.join(save_dir, f"pca_feature_contrib_top{k}.png"),
                     dpi=120, bbox_inches="tight")
    plt.show() if show else plt.close(fig4)

    return dict(
        pca=pca,
        mu=mu,
        sigma=sigma,
        scores=Z,
        evr=evr,
        components=comps,
        importance=importance,                 # mean-normalized over ALL PCs
        importance_topk=totals,               # raw top-k (evr-weighted if chosen)
        contrib_topk_per_pc=contrib_k,        # (k, D)
        feature_names=names,
        topk=k,
        weight_by_evr=bool(weight_by_evr),
    )

def select_top_k_features(importance: np.ndarray, k: int = 5) -> np.ndarray:
    """Return indices of the top-k most important features (descending)."""
    k = min(k, importance.shape[0])
    return np.argsort(importance)[::-1][:k]

# ---------- 3) Env wrappers ----------

class TopKFeatureMask(gym.ObservationWrapper):
    """
    Gym wrapper that keeps only the selected feature indices.
    Works for Box spaces with last-dim = features.
    """
    def __init__(self, env: gym.Env, keep_indices: Sequence[int]):
        super().__init__(env)
        self.keep = np.asarray(keep_indices, dtype=int)
        assert isinstance(env.observation_space, spaces.Box), "TopKFeatureMask expects Box observation space"
        old: spaces.Box = env.observation_space
        assert old.shape is not None and old.shape[-1] >= len(self.keep)
        low = _to_numpy(old.low)[..., self.keep]
        high = _to_numpy(old.high)[..., self.keep]
        self.observation_space = spaces.Box(low=low, high=high, dtype=old.dtype)

    def observation(self, observation):
        arr = _to_numpy(observation)
        # support (D,) or (..., D)
        return arr[..., self.keep]

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper

class PCAObservationWrapper(VecEnvWrapper):
    def __init__(self, venv, mu, sigma, components, n_components=2, dtype=np.float32):
        super().__init__(venv)
        D = venv.observation_space.shape[0]
        self.mu = np.asarray(mu, np.float32).reshape(1, D)
        self.sigma = np.maximum(np.asarray(sigma, np.float32).reshape(1, D), 1e-8)
        W = np.asarray(components, np.float32)
        assert W.shape[1] == D and n_components <= W.shape[0]
        self.W = W[:n_components, :]                          # (n_comp, D)
        self.dtype = dtype
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_components,), dtype=dtype
        )
        self._last_raw_obs = None

    def _project(self, obs):
        x = np.asarray(obs, np.float32)                       # (n_envs, D)
        xc = (x - self.mu) / self.sigma
        z = xc @ self.W.T                                     # (n_envs, n_comp)
        return z.astype(self.dtype, copy=False)

    # def reset(self):
    #     return self._project(self.venv.reset())

    # def step_wait(self):
    #     obs, rews, dones, infos = self.venv.step_wait()
    #     return self._project(obs), rews, dones, infos
    def reset(self):
        raw = self.venv.reset()
        self._last_raw_obs = np.asarray(raw, np.float32).copy()
        return self._project(raw)

    def step_wait(self):
        raw, rews, dones, infos = self.venv.step_wait()
        raw = np.asarray(raw, np.float32)
        for i in range(len(infos)):
            infos[i] = dict(infos[i])
            infos[i]["raw_obs"] = raw[i].copy()
            if dones[i]:
                infos[i]["raw_obs_terminal"] = raw[i].copy()
        return self._project(raw), rews, dones, infos


# ---------- 4) Example usage ----------
def train_model(env, args):
    from stable_baselines3 import PPO
    from utils import FEATURE_NAMES
        
    model = PPO(policy="MlpPolicy",
                     env=env,
                     n_steps=2048,
                     #batch_size=n_steps*3,
                     learning_rate=0.00003,
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
                     tensorboard_log='./output_malota/pca')
    model.learn(total_timesteps=1_000)
        # 1) collect data
    X = collect_observations(model, env, n_steps=10_000, deterministic=True)
        # 2) analyze
    feature_names = FEATURE_NAMES
    results = pca_analysis(X, n_components=7, feature_names=feature_names, show=True, save_dir="pca_reports")
    # 3) pick top-5 features
    top5_idx = select_top_k_features(results["importance"], k=5)
    print("Top-5 features:", [results["feature_names"][i] for i in top5_idx])
    # 4a) wrap env to mask to top-5 (train a new policy on this)
    #env_top5 = TopKFeatureMask(env, keep_indices=top5_idx)
    # 4b) OR wrap env to project to 2 or 3 PCs (train a new policy on this)
    #env_pc2 = PCAObservationWrapper(env, results["mu"], results["sigma"], results["components"], n_components=2)
    env = PCAObservationWrapper(env, results["mu"], results["sigma"], results["components"], n_components=5) #should keep even 5
    #initializing model again, so that it learns on 3 inp features env
    model = PPO(policy="MlpPolicy",
                     env=env,
                     n_steps=2048,
                     #batch_size=n_steps*3,
                     learning_rate=0.00003,
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
                     tensorboard_log='./output_malota/pca')
    print('Training model inside PCA module')
    #model.learn(total_timesteps=10_000)
    return model, env, results
