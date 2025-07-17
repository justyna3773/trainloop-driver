import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import torch as th
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gym



# ====================== PCA-BASED FEATURE SELECTION FUNCTIONS ======================
def select_features_by_components(components: np.ndarray, 
                                 threshold: float, 
                                 method: str = 'max_loading'):
    """
    Selects feature indices based on PCA loadings using specified method and threshold.
    
    Args:
        components: PCA components array (n_components x n_features)
        threshold: Loading threshold value for feature selection
        method: Selection method ('max_loading' or 'variance_contribution')
    
    Returns:
        List of selected feature indices
    """
    n_features = components.shape[1]
    selected_indices = []

    if method == 'max_loading':
        # Select features where max absolute loading >= threshold
        for j in range(n_features):
            if np.max(np.abs(components[:, j])) >= threshold:
                selected_indices.append(j)
                
    elif method == 'variance_contribution':
        # Select features where sum of squared loadings >= threshold
        for j in range(n_features):
            contribution = np.sum(components[:, j] ** 2)
            if contribution >= threshold:
                selected_indices.append(j)
    else:
        raise ValueError(f"Invalid method: {method}. Use 'max_loading' or 'variance_contribution'")
    
    return selected_indices






class TruePCAFeatureExtractor(BaseFeaturesExtractor):
    """Applies precomputed PCA transformation as fixed preprocessing layer."""
    def __init__(self, observation_space: gym.Space, pca_components: np.ndarray):
        self.obs_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=4)
        
        # Convert PCA components to PyTorch layer
        self.pca_layer = nn.Linear(self.obs_dim, 4, bias=False)
        
        # Set weights from PCA and freeze
        with th.no_grad():
            self.pca_layer.weight.copy_(th.from_numpy(pca_components).float())
        self.pca_layer.weight.requires_grad_(False)  # Make non-trainable

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.pca_layer(obs)

class PCAFeaturePolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TruePCAFeatureExtractor,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            activation_fn=nn.ReLU,
            **kwargs
        )

# --------------------------------------------------
# Usage example with PCA precomputation on environment data
# --------------------------------------------------

def compute_pca(env: gym.Env, n_samples: int = 1000, visualize=False, n_components=4) -> np.ndarray:
    """Collects environment observations and computes PCA components."""
    is_vec_env = hasattr(env, "num_envs")
    n_envs = env.num_envs if is_vec_env else 1
    
    observations = []
    obs = env.reset()
    total_collected = 0
    
    while total_collected < n_samples:
        if is_vec_env:
            # Convert actions to native Python ints
            actions = [int(env.action_space.sample()) for _ in range(n_envs)]  # Convert to int
            obs, _, dones, _ = env.step(actions)  # Pass as list of native ints
            
            observations.extend(obs)
            total_collected += len(obs)
        else:
            # Convert single action to native int
            action = int(env.action_space.sample())  # Convert to int
            obs, _, done, _ = env.step(action)
            observations.append(obs)
            total_collected += 1
            if done:
                obs = env.reset()

        if total_collected > n_samples * 2:
            break

    observations = np.array(observations[:n_samples])
    pca = PCA(n_components=n_components)
    import joblib
    pca.fit(observations)
    #np.save('pca_components.npy', pca.components_)
    joblib.dump(pca, 'rl_pca_model.joblib')

    if visualize:
        return pca.components_, pca, observations
    return pca.components_

def visualize_pca(pca: PCA, observations: np.ndarray, feature_names: list = None):
    """Visualizes PCA results with feature importance."""
    plt.figure(figsize=(15, 6))
    
    # 1. Explained Variance Ratio
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
             np.cumsum(pca.explained_variance_ratio_), 'r-')
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Explained')
    plt.title('Explained Variance Ratio')
    plt.legend(['Cumulative', 'Individual'])

    # 2. Component Loadings Heatmap
    plt.subplot(1, 2, 2)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(observations.shape[1])]
        
    sns.heatmap(loadings, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm',
                xticklabels=[f'PC{i+1}' for i in range(pca.n_components_)],
                yticklabels=feature_names)
    plt.title('PCA Component Loadings')
    plt.xlabel('Principal Components')
    plt.ylabel('Original Features')
    plt.tight_layout()
    plt.show()

    # 3. Biplot (combination of scores and loadings)
    plt.figure(figsize=(12, 8))
    scores = pca.transform(observations)
    scale = 1.0/(scores[:, :2].max() - scores[:, :2].min())
    
    plt.scatter(scores[:, 0]*scale, scores[:, 1]*scale, alpha=0.2)
    
    # Plot feature arrows
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, pca.components_[0, i], 
                  pca.components_[1, i], 
                  color='r', alpha=0.5)
        plt.text(pca.components_[0, i]*1.15, 
                 pca.components_[1, i]*1.15, 
                 feature, color='r')
    
    plt.xlabel('PC1 (%.1f%%)' % (pca.explained_variance_ratio_[0]*100))
    plt.ylabel('PC2 (%.1f%%)' % (pca.explained_variance_ratio_[1]*100))
    plt.title('PCA Biplot')
    plt.grid()
    plt.show()
