# import torch
# import torch.nn as nn
# import gym
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from captum.attr import IntegratedGradients
# from stable_baselines3 import PPO
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.vec_env import DummyVecEnv

# # 1. CNN Feature Extractor with Visualization Capabilities
# class CNNFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.Space, 
#                  features_dim: int = 64,
#                  encoder_channels: list = [16, 32]):
#         super().__init__(observation_space, features_dim)
#         self.input_shape = observation_space.shape
#         c, h, w = self.input_shape
        
#         # CNN layers
#         self.conv = nn.Sequential(
#             nn.Conv2d(c, encoder_channels[0], kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten()
#         )
        
#         # Calculate CNN output size
#         with torch.no_grad():
#             sample = torch.zeros(1, *self.input_shape)
#             n_flatten = self.conv(sample).shape[1]
        
#         # Projection to feature space
#         self.projection = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU()
#         )
#         self._features_dim = features_dim

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         return self.projection(self.conv(observations))
    
#     def compute_feature_importance(self, observations, n_steps=50):
#         """
#         Compute feature importance using Integrated Gradients
#         Returns importance matrix of shape [features_dim, 15, 7]
#         """
#         self.eval()
        
#         # Convert to tensor if needed
#         if not isinstance(observations, torch.Tensor):
#             obs_tensor = torch.tensor(observations, dtype=torch.float32)
#         else:
#             obs_tensor = observations
        
#         # Create baseline (10th percentile)
#         baseline = torch.quantile(obs_tensor, 0.1, dim=0, keepdim=True)
        
#         # Initialize importance matrix
#         importance_matrix = np.zeros((self._features_dim, *obs_tensor.shape[2:]))
        
#         # Compute importance for each feature dimension
#         for feature_idx in range(self._features_dim):
#             def model_forward(x):
#                 return self(x)[:, feature_idx]
            
#             ig = IntegratedGradients(model_forward)
#             attr = ig.attribute(obs_tensor, 
#                                 baselines=baseline,
#                                 n_steps=n_steps)
            
#             # Average absolute attribution across samples
#             avg_attr = torch.mean(torch.abs(attr), dim=0)
#             importance_matrix[feature_idx] = avg_attr.squeeze(0).detach().cpu().numpy()
        
#         return importance_matrix

#     def visualize_feature_importance(self, observations, feature_names=None, time_labels=None):
#         """
#         Visualize average feature importance across all latent dimensions
#         """
#         # Compute importance
#         importance_matrix = self.compute_feature_importance(observations)
        
#         # Average across feature dimensions
#         avg_importance = np.mean(importance_matrix, axis=0)
        
#         # Create figure
#         plt.figure(figsize=(12, 8))
        
#         # Default labels
#         if feature_names is None:
#             feature_names = [f"Feature {i}" for i in range(avg_importance.shape[1])]
#         if time_labels is None:
#             time_labels = [f"T-{i}" for i in range(avg_importance.shape[0]-1, -1, -1)]
        
#         # Plot heatmap
#         ax = sns.heatmap(avg_importance, 
#                          annot=True, 
#                          fmt=".2f",
#                          cmap="viridis",
#                          yticklabels=time_labels,
#                          xticklabels=feature_names,
#                          linewidths=0.5)
        
#         plt.title("Average Feature Importance Across Latent Dimensions", fontsize=16)
#         plt.xlabel("Features", fontsize=14)
#         plt.ylabel("Time Steps", fontsize=14)
#         plt.xticks(rotation=45, ha='right')
#         plt.yticks(rotation=0)
        
#         # Add colorbar
#         cbar = ax.collections[0].colorbar
#         cbar.set_label('Feature Importance', fontsize=12)
        
#         plt.tight_layout()
#         plt.show()
        
#         return avg_importance

# # 2. MLP Policy with Pretrained CNN Features
# class CNNMLPFeatureExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.Space, 
#                  cnn_extractor: nn.Module,
#                  mlp_dims: list = [128, 64],
#                  features_dim: int = 64):
#         super().__init__(observation_space, features_dim)
        
#         # Store and freeze CNN extractor
#         self.cnn_extractor = cnn_extractor
#         for param in self.cnn_extractor.parameters():
#             param.requires_grad = False
        
#         # Build MLP
#         mlp_layers = []
#         prev_dim = features_dim
#         for dim in mlp_dims:
#             mlp_layers.append(nn.Linear(prev_dim, dim))
#             mlp_layers.append(nn.ReLU())
#             prev_dim = dim
#         self.mlp = nn.Sequential(*mlp_layers)
        
#         # Final output layer
#         self.output_layer = nn.Linear(mlp_dims[-1], features_dim)
#         self._features_dim = features_dim

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         # CNN feature extraction
#         with torch.no_grad():
#             cnn_features = self.cnn_extractor(observations)
        
#         # MLP processing
#         mlp_output = self.mlp(cnn_features)
#         return self.output_layer(mlp_output)

# # 3. Main training workflow with visualization
# def train_model(env):

    
#     # Feature names for visualization
#     FEATURE_NAMES = [
#         "vmAllocatedRatio",
#         "avgPUUtilization",
#         "avgMemoryUtilization",
#         "p90MemoryUtilization",
#         "p90CPUUtilization",
#         "waitingJobsRatioGlobal",
#         "waitingJobsRatioRecent"
#     ]
#     TIME_LABELS = [f"T-{i}" for i in range(14, -1, -1)]  # T-14 (oldest) to T-0 (current)
    
#     # Phase 1: Pretrain CNN feature extractor
#     print("=== Phase 1: Pretraining CNN Feature Extractor ===")
#     model_cnn = PPO(
#         "CnnPolicy",
#         env,
#         policy_kwargs={
#             "features_extractor_class": CNNFeatureExtractor,
#             "features_extractor_kwargs": {
#                 "features_dim": 64,
#                 "encoder_channels": [16, 32]
#             },
#             "net_arch": [dict(pi=[64], vf=[64])]
#         },
#         verbose=1
#     )
#     model_cnn.learn(total_timesteps=10000)
    
#     # Extract CNN feature extractor
#     cnn_extractor = model_cnn.policy.features_extractor
    
#     # Visualize feature importance after pretraining
#     print("\n=== Visualizing CNN Feature Importance ===")
#     sample_obs = collect_observations(env, num_samples=100)
#     cnn_extractor.visualize_feature_importance(
#         sample_obs,
#         feature_names=FEATURE_NAMES,
#         time_labels=TIME_LABELS
#     )
    
#     # Phase 2: Train MLP policy with pretrained CNN features
#     print("\n=== Phase 2: Training MLP Policy with CNN Features ===")
#     model_mlp = PPO(
#         "MlpPolicy",
#         env,
#         policy_kwargs={
#             "features_extractor_class": CNNMLPFeatureExtractor,
#             "features_extractor_kwargs": {
#                 "cnn_extractor": cnn_extractor,
#                 "mlp_dims": [128, 64],
#                 "features_dim": 64
#             },
#             "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
#         },
#         verbose=1
#     )
#     #model_mlp.learn(total_timesteps=100000)
    
#     return model_mlp

# def collect_observations(env, num_samples=10):
#     """Collect sample observations from environment"""
#     observations = []
#     obs = env.reset()
#     for _ in range(num_samples):
#         observations.append(obs)
#         action = [env.action_space.sample() for _ in range(env.num_envs)]
#         obs, _, _, _ = env.step(action)
#     return np.concatenate(observations)

# # # Run the training
# # if __name__ == "__main__":
# #     trained_model = train_model()
# #     trained_model.save("ppo_cnn_mlp_model")

import torch
import torch.nn as nn
import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from captum.attr import IntegratedGradients
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

# 1. CNN Feature Extractor with Visualization Capabilities
class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, 
                 features_dim: int = 64,
                 encoder_channels: list = [16, 32]):
        super().__init__(observation_space, features_dim)
        self.input_shape = observation_space.shape
        c, h, w = self.input_shape
        
        # CNN layers
        self.conv1 = nn.Conv2d(c, encoder_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=3, stride=1, padding=1)
        self.conv = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, *self.input_shape)
            n_flatten = self.conv(sample).shape[1]
        
        # Projection to feature space
        self.projection = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.conv(observations))
    
    def compute_feature_importance(self, observations, n_steps=50):
        """Compute feature importance using Integrated Gradients"""
        self.eval()
        
        # Convert to tensor if needed
        if not isinstance(observations, torch.Tensor):
            obs_tensor = torch.tensor(observations, dtype=torch.float32)
        else:
            obs_tensor = observations
        
        # Create baseline (10th percentile)
        baseline = torch.quantile(obs_tensor, 0.1, dim=0, keepdim=True)
        
        # Initialize importance matrix
        importance_matrix = np.zeros((self._features_dim, *obs_tensor.shape[2:]))
        
        # Compute importance for each feature dimension
        for feature_idx in range(self._features_dim):
            def model_forward(x):
                return self(x)[:, feature_idx]
            
            ig = IntegratedGradients(model_forward)
            attr = ig.attribute(obs_tensor, 
                                baselines=baseline,
                                n_steps=n_steps)
            
            # Average absolute attribution across samples
            avg_attr = torch.mean(torch.abs(attr), dim=0)
            importance_matrix[feature_idx] = avg_attr.squeeze(0).detach().cpu().numpy()
        
        return importance_matrix

    def compute_average_feature_importance(self, observations):
        """Compute and return average feature importance heatmap data"""
        importance_matrix = self.compute_feature_importance(observations)
        return np.mean(importance_matrix, axis=0)

# 2. MLP Policy with Pretrained CNN Features
class CNNMLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, 
                 cnn_extractor: nn.Module,
                 mlp_dims: list = [128, 64],
                 features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        # Store and freeze CNN extractor
        self.cnn_extractor = cnn_extractor
        for param in self.cnn_extractor.parameters():
            param.requires_grad = False
        
        # Build MLP
        mlp_layers = []
        prev_dim = features_dim
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.ReLU())
            prev_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final output layer
        self.output_layer = nn.Linear(mlp_dims[-1], features_dim)
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        with torch.no_grad():
            cnn_features = self.cnn_extractor(observations)
        
        # MLP processing
        mlp_output = self.mlp(cnn_features)
        return self.output_layer(mlp_output)

# 3. Callback for TensorBoard logging of feature importance heatmaps
class FeatureImportanceHeatmapCallback(BaseCallback):
    def __init__(self, 
                 tb_writer: SummaryWriter, 
                 feature_names: list,
                 time_labels: list,
                 log_freq: int = 1000,
                 num_samples: int = 10,
                 verbose: int = 0):
        super().__init__(verbose)
        self.tb_writer = tb_writer
        self.feature_names = feature_names
        self.time_labels = time_labels
        self.log_freq = log_freq
        self.num_samples = num_samples

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Get feature extractor
            fe = self.model.policy.features_extractor
            
            if hasattr(fe, 'compute_average_feature_importance'):
                # Collect observations
                observations = collect_observations(self.training_env, self.num_samples)
                
                # Compute average feature importance
                avg_importance = fe.compute_average_feature_importance(observations)
                
                # Create heatmap figure
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(avg_importance, 
                            ax=ax,
                            annot=True, 
                            fmt=".2f",
                            cmap="viridis",
                            yticklabels=self.time_labels,
                            xticklabels=self.feature_names,
                            linewidths=0.5)
                
                ax.set_title(f"Feature Importance at Step {self.num_timesteps}", fontsize=16)
                ax.set_xlabel("Features", fontsize=14)
                ax.set_ylabel("Time Steps", fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                # Log to TensorBoard
                self.tb_writer.add_figure(
                    "feature_importance/heatmap",
                    fig,
                    self.num_timesteps
                )
                plt.close(fig)
                
        return True

# 4. Main training workflow with visualization
def train_model(env):
    # Feature names for visualization
    FEATURE_NAMES = [
        "vmAllocatedRatio",
        "avgPUUtilization",
        "avgMemoryUtilization",
        "p90MemoryUtilization",
        "p90CPUUtilization",
        "waitingJobsRatioGlobal",
        "waitingJobsRatioRecent"
    ]
    TIME_LABELS = [f"T-{i}" for i in range(14, -1, -1)]  # T-14 (oldest) to T-0 (current)
    
    # Create TensorBoard writer
    tb_writer = SummaryWriter(log_dir="logs/feature_importance")
    
    # Initialize callback
    heatmap_callback = FeatureImportanceHeatmapCallback(
        tb_writer=tb_writer,
        feature_names=FEATURE_NAMES,
        time_labels=TIME_LABELS,
        log_freq=1000,
        num_samples=10
    )
    
    # Phase 1: Pretrain CNN feature extractor
    print("=== Phase 1: Pretraining CNN Feature Extractor ===")
    model_cnn = PPO(
        "CnnPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": CNNFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": 64,
                "encoder_channels": [16, 32]
            },
            "net_arch": [dict(pi=[64], vf=[64])]
        },
        verbose=1,
        tensorboard_log="logs/ppo_tensorboard"
    )
    model_cnn.learn(total_timesteps=10000, callback=heatmap_callback)
    
    # Close TensorBoard writer
    tb_writer.close()
    
    # Extract CNN feature extractor
    cnn_extractor = model_cnn.policy.features_extractor
    
    # Phase 2: Train MLP policy with pretrained CNN features
    print("\n=== Phase 2: Training MLP Policy with CNN Features ===")
    model_mlp = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": CNNMLPFeatureExtractor,
            "features_extractor_kwargs": {
                "cnn_extractor": cnn_extractor,
                "mlp_dims": [128, 64],
                "features_dim": 64
            },
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
        },
        verbose=1,
        tensorboard_log="logs/ppo_tensorboard"
    )
    model_mlp.learn(total_timesteps=100000)
    
    return model_mlp

def collect_observations(env, num_samples=10):
    """Collect sample observations from environment"""
    observations = []
    obs = env.reset()
    for _ in range(num_samples):
        observations.append(obs)
        action = [env.action_space.sample() for _ in range(env.num_envs)]
        obs, _, _, _ = env.step(action)
    return np.concatenate(observations)

# if __name__ == "__main__":
#     # Create environment (replace with your actual environment)
#     env = gym.make("CartPole-v1")
#     env = DummyVecEnv([lambda: env])
    
#     trained_model = train_model(env)
#     trained_model.save("ppo_cnn_mlp_model")