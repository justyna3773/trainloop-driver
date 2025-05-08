import gym
import numpy as np
import torch as th
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from typing import Dict, List, Tuple, Type, Union, Callable
import json

import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from typing import Callable

import torch as th
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gym

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

def compute_pca(env: gym.Env, n_samples: int = 1000, visualize=False) -> np.ndarray:
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
    pca = PCA(n_components=4)
    pca.fit(observations)
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



# Step 4: Verify PCA layer is fixed


class FeatureAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=4)
        self.obs_dim = observation_space.shape[0]
        self.attention = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.obs_dim * 4))
        # Register buffer to store attention weights
        self.register_buffer("latest_attn_weights", th.zeros(4, self.obs_dim))  # Shape: (4, obs_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.shape[0]
        attn_params = self.attention(obs)
        attn_weights = attn_params.view(batch_size, self.obs_dim, 4)
        attn_weights = th.softmax(attn_weights, dim=1)
        
        # Save the latest weights (average over batch)
        self.latest_attn_weights = attn_weights.mean(dim=0).detach().cpu().permute(1, 0)  # (4, obs_dim)
        
        attended_features = th.bmm(obs.unsqueeze(1), attn_weights).squeeze(1)
        return attended_features
class FeatureAttentionExtractor_backup(BaseFeaturesExtractor):
    """Reduces input features to 4 through attention-weighted combinations."""
    def __init__(self, observation_space: gym.Space):
        self.obs_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=4)  # Output 4 features
        
        # Attention network outputs parameters for 4 weight vectors
        self.attention = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.obs_dim * 4)  # Output enough parameters for 4 weight vectors
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.shape[0]
        # Generate attention parameters (batch_size, obs_dim*4)
        attn_params = self.attention(obs)
        # Reshape into (batch_size, obs_dim, 4) - 4 weight vectors per observation
        attn_weights = attn_params.view(batch_size, self.obs_dim, 4)
        # Apply softmax along feature dimension to get normalized weights per vector
        attn_weights = th.softmax(attn_weights, dim=1)
        
        # Compute weighted sum for each of the 4 vectors (batch matrix multiplication)
        # obs shape: (batch_size, 1, obs_dim), attn_weights: (batch_size, obs_dim, 4)
        # Result shape: (batch_size, 1, 4) -> squeezed to (batch_size, 4)
        attended_features = th.bmm(obs.unsqueeze(1), attn_weights).squeeze(1)
        return attended_features
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class AttentionWeightsCallback_backup(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.attn_weights_history = []

    def _on_step(self) -> bool:
        # Log every 100 steps
        if self.n_calls % 100 == 0:
            # Access the feature extractor from the policy
            feature_extractor = self.model.policy.features_extractor
            if hasattr(feature_extractor, 'latest_attn_weights'):
                weights = feature_extractor.latest_attn_weights.numpy()
                self.logger.record("attention/weights", np.mean(weights))
                self.attn_weights_history.append(weights)
        return True
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class FeatureAttentionWeightsCallback(BaseCallback):
    """Logs attention weights for all features and heads in a structured way."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.attn_weights_history = []

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:  # Adjust logging frequency as needed
            feature_extractor = self.model.policy.features_extractor
            if hasattr(feature_extractor, 'latest_attn_weights'):
                # Shape: (num_heads, obs_dim)
                weights = feature_extractor.latest_attn_weights.numpy()
                self.attn_weights_history.append({
                    "step": self.num_timesteps,
                    "weights": weights
                })
        return True
import torch as th
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FeatureAttentionExtractor2D(BaseFeaturesExtractor):
    """Reduces input features to 2 dimensions using attention mechanism"""
    def __init__(self, observation_space: gym.Space):
        self.obs_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=2)
        
        # Attention network outputs parameters for 2 weight vectors
        self.attention = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.obs_dim * 2)  # Output parameters for 2 weight vectors
        )
        
        # Register buffer to store attention weights for 2 heads
        self.register_buffer("latest_attn_weights", th.zeros(2, self.obs_dim))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        batch_size = obs.shape[0]
        
        # Generate attention parameters (batch_size, obs_dim*2)
        attn_params = self.attention(obs)
        
        # Reshape to (batch_size, obs_dim, 2)
        attn_weights = attn_params.view(batch_size, self.obs_dim, 2)
        
        # Apply softmax along feature dimension (dim=1)
        attn_weights = th.softmax(attn_weights, dim=1)
        
        # Save latest weights (average over batch)
        self.latest_attn_weights = attn_weights.mean(dim=0).detach().permute(1, 0)  # (2, obs_dim)
        
        # Compute weighted features (batch_size, 2)
        attended_features = th.bmm(obs.unsqueeze(1), attn_weights).squeeze(1)
        
        return attended_features
class AttentionPolicy2D(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=FeatureAttentionExtractor2D,
            features_extractor_kwargs={},
            net_arch=[dict(pi=[64, 64], vf=[64, 64])]
        )
class FeatureAttentionPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable,
        **kwargs
    ):
        # super().__init__(
        #     observation_space,
        #     action_space,
        #     lr_schedule,
        #     features_extractor_class=FeatureAttentionExtractor,
        #     net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        #     activation_fn=nn.ReLU,
        #     **kwargs
        # )
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=FeatureAttentionExtractor,
            features_extractor_kwargs={},
            # Smaller MLP layers after attention
            net_arch=[
                dict(pi=[16,], vf=[16,])  # Reduced from [64, 64]
            ],
            activation_fn=nn.ReLU,
            **kwargs
        )
import torch as th
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


        

# Register policy
class FeatureAttentionExtractor_backup(BaseFeaturesExtractor):
    """Focuses attention on specific features of the current observation"""
    def __init__(self, observation_space: gym.Space):
        self.obs_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=self.obs_dim * 2)
        
        # Feature attention network
        self.attention = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.obs_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # Compute attention weights for each feature
        weights = self.attention(obs)
        # Apply attention weights to features
        attended_features = obs * weights
        # Concatenate with original features
        return th.cat([attended_features, obs], dim=-1)

class FeatureAttentionPolicy_backup(ActorCriticPolicy):
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
            features_extractor_class=FeatureAttentionExtractor,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            activation_fn=nn.ReLU,
            **kwargs
        )

PPO.policy_aliases["FeatureAttentionPolicy"] = FeatureAttentionPolicy



########### ATTENTION VISUALIZATION FUNCTIONS ###############

def get_attention_weights(model, observation: np.ndarray) -> np.ndarray:
    """
    Returns attention weights for a given observation.
    
    Args:
        model: Trained RL model (PPO, A2C, etc.)
        observation: Input observation array
    
    Returns:
        Attention weights of shape (obs_dim, 4)
    """
    # Convert observation to tensor
    obs_tensor = th.as_tensor(observation).float().to(model.device)
    # Add batch dimension if needed
    if len(obs_tensor.shape) == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    
    # Get feature extractor from the policy
    feature_extractor = model.policy.features_extractor
    
    with th.no_grad():
        # Forward pass through the attention network
        attn_params = feature_extractor.attention(obs_tensor)
        # Reshape and apply softmax
        batch_size = obs_tensor.size(0)
        attn_weights = attn_params.view(batch_size, feature_extractor.obs_dim, 4)
        attn_weights = th.softmax(attn_weights, dim=1)
    
    # Remove batch dimension if single observation
    return attn_weights.squeeze().cpu().numpy()

def print_top_features(attn_weights: np.ndarray, feature_names: list = None, top_k: int = 3):
    """
    Prints top-k important features for each output dimension.
    
    Args:
        attn_weights: Attention weights from get_attention_weights()
        feature_names: Names of input features (optional)
        top_k: Number of top features to display
    """
    n_outputs = attn_weights.shape[1]
    
    for output_idx in range(n_outputs):
        weights = attn_weights[:, output_idx]
        sorted_indices = np.argsort(weights)[::-1]  # Descending order
        
        # Get top-k indices and weights
        top_indices = sorted_indices[:top_k]
        top_weights = weights[top_indices]
        
        # Use feature names if available
        if feature_names:
            top_features = [feature_names[i] for i in top_indices]
        else:
            top_features = top_indices
        
        print(f"Output feature {output_idx + 1} top-{top_k} input features:")
        feature_mapping = {0: "vmAllocatedRatio",
        1: "avgCPUUtilization",
        2: "p90CPUUtilization",
        3: "avgMemoryUtilization",
        4: "p90MemoryUtilization",
        5: "waitingJobsRatioGlobal",
        6: "waitingJobsRatioRecent"}
        for feat, weight in zip(top_features, top_weights):
            print(f"  Feature {feature_mapping[feat]}: {weight:.3f}")
        
        print()