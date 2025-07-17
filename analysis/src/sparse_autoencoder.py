import torch
import torch.nn as nn
import gym
import torch.nn.functional as F
import numpy as np
from collections import deque
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients

class AdaptiveSparseAE(nn.Module):
    def __init__(self, input_dim, max_latent_dim, hidden_dims=[64, 32], 
                 sparsity_coeff=0.5, temp=1.0, threshold=0.5):
        super().__init__()
        self.max_latent_dim = max_latent_dim
        self.threshold = threshold
        self.temp = temp
        self.encoder_output_dim = hidden_dims[-1]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Projection to max_latent_dim
        self.projection = nn.Linear(self.encoder_output_dim, max_latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = max_latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Sparsity control
        self.sparsity_coeff = sparsity_coeff
        self.active_dims = max_latent_dim
        
        # Permanent mask for inactive dimensions
        self.register_buffer('permanent_mask', torch.ones(max_latent_dim, dtype=torch.bool))
        
    def forward(self, x):
        # Encode features
        features = self.encoder(x)
        projected = self.projection(features)
        
        # Gumbel-Softmask for differentiable selection
        if self.training:
            uniforms = torch.rand_like(projected)
            gumbels = -torch.log(-torch.log(uniforms))
            noisy_logits = (projected + gumbels) / self.temp
            mask = torch.sigmoid(noisy_logits)
        else:
            mask = (torch.sigmoid(projected) > self.threshold).float()
        
        # Apply permanent mask to disable pruned dimensions
        mask = mask * self.permanent_mask.reshape(1, -1)
        
        # Apply mask to projected features
        latent = mask * projected
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Sparsity regularization
        sparsity_loss = torch.mean(torch.sigmoid(projected)) * self.sparsity_coeff
        
        # Track active dimensions
        self.active_dims = torch.mean((mask > self.threshold).float()).item() * self.max_latent_dim
        
        return reconstructed, latent, sparsity_loss

    def prune_inactive_dimensions(self, observations, threshold=0.4):
        """Permanently disable unused latent dimensions"""
        self.eval()
        with torch.no_grad():
            # Compute mask for all observations
            features = self.encoder(observations)
            projected = self.projection(features)
            mask = torch.sigmoid(projected) > self.threshold
            
            # Calculate activity fractions
            fraction_active = mask.float().mean(dim=0)
            
            # Identify dimensions active less than threshold
            inactive_dims = torch.where(fraction_active < threshold)[0].tolist()
            active_mask = fraction_active >= threshold
            
            # Update permanent mask
            self.permanent_mask = active_mask
            
            # Zero out weights for inactive dimensions
            for dim in inactive_dims:
                self.projection.weight.data[dim] = 0
                self.projection.bias.data[dim] = 0
                self.decoder[0].weight.data[:, dim] = 0
            
            # Get active dimension count
            active_count = active_mask.sum().item()
        
        print(f"Pruned {len(inactive_dims)} inactive dimensions. "
              f"Active dimensions: {active_count}/{self.max_latent_dim}")
        return inactive_dims


class AdaptiveSparseAE_FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, 
                 max_latent_dim=8,
                 hidden_dims=[64, 32],
                 sparsity_coeff=0.05,
                 temp=0.5,
                 threshold=0.2, pretrained_ae=None):
        input_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=max_latent_dim)
        if pretrained_ae is not None:
            self.autoencoder = pretrained_ae
        else: 
            self.autoencoder = AdaptiveSparseAE(
                input_dim,
                max_latent_dim,
                hidden_dims,
                sparsity_coeff,
                temp,
                threshold
            )
        self.active_dims = max_latent_dim
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.autoencoder.encoder(observations)
            projected = self.autoencoder.projection(features)
            
            # Apply both threshold and permanent mask
            mask = (torch.sigmoid(projected) > self.autoencoder.threshold)
            mask = mask * self.autoencoder.permanent_mask.reshape(1, -1)
            return mask * projected
            
    def train_autoencoder(self, observations: torch.Tensor):
        self.autoencoder.train()
        reconstructed, latent, sparsity_loss = self.autoencoder(observations)
        
        # Reconstruction loss
        recon_loss = nn.MSELoss()(reconstructed, observations)
        
        # Total loss
        total_loss = recon_loss + sparsity_loss
        
        # Backpropagation
        self.autoencoder.zero_grad()
        total_loss.backward()
        
        # Update active dimensions count
        self.active_dims = self.autoencoder.active_dims
        
        return total_loss.item()
    def compute_feature_importance(self, observations, n_steps=50):
        # Ensure autoencoder is in eval mode
        self.autoencoder.eval()
        
        # Convert to tensor
        obs_tensor = torch.tensor(observations, dtype=torch.float32)
        obs_tensor = obs_tensor.squeeze(1)
        # Create baseline (10th percentile of features)
        baseline = torch.quantile(obs_tensor, 0.1, dim=0) * torch.ones_like(obs_tensor)
        
        # Get active dimensions from permanent mask
        active_dims = torch.where(self.autoencoder.permanent_mask)[0].tolist()
        
        # Initialize importance matrix only for active dimensions
        importance_matrix = np.zeros((len(active_dims), obs_tensor.shape[1]))
        
        # Compute importance only for active dimensions
        for idx, dim in enumerate(active_dims):
            def model_forward(x):
                # Get projected features for specific dimension
                features = self.autoencoder.encoder(x)
                projected = self.autoencoder.projection(features)
                return projected[:, dim]
            
            ig = IntegratedGradients(model_forward)
            attr = ig.attribute(obs_tensor, 
                            baselines=baseline,
                            target=None,
                            n_steps=n_steps)
            
            # Average absolute attribution across samples
            importance_matrix[idx] = torch.mean(torch.abs(attr), dim=0).detach().numpy()
        
        return importance_matrix, active_dims

    def visualize_feature_importance(self, importance_matrix, active_dims, feature_names=None):
        """
        Visualize feature importance as a heatmap
        """
        plt.figure(figsize=(12, 8))
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(importance_matrix.shape[1])]
        
        # Create latent dimension names using active dim indices
        latent_names = [f"Latent Dim {dim}" for dim in active_dims]
        
        # Plot heatmap
        sns.heatmap(importance_matrix, 
                    annot=True, 
                    fmt=".2f",
                    cmap="viridis",
                    xticklabels=feature_names,
                    yticklabels=latent_names)
        
        plt.title("Feature Importance in Active Latent Dimensions")
        plt.xlabel("Input Features")
        plt.ylabel("Latent Dimensions")
        plt.tight_layout()
        plt.show()

class AdaptiveAETrainer(BaseCallback):
    def __init__(self, train_freq=1000, batch_size=256, buffer_size=10000, verbose=0):
        super().__init__(verbose)
        
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.active_dim_history = []
        
    def _on_step(self) -> bool:
        self.buffer.append(self.locals["obs_tensor"].cpu().numpy().copy())
        
        if self.n_calls % self.train_freq == 0 and len(self.buffer) >= self.batch_size:
            self.train_autoencoder()
            
        # Log dimension info every 100 steps
        if self.n_calls % 100 == 0:
            fe = self.model.policy.features_extractor
            active_dims = fe.active_dims
            self.active_dim_history.append(active_dims)
            self.logger.record("ae/active_dims", active_dims)
            
        return True
    
    def train_autoencoder(self):
        batch_idxs = np.random.choice(len(self.buffer), self.batch_size)
        obs_batch = torch.tensor(
            np.array([self.buffer[i] for i in batch_idxs]),
            dtype=torch.float32,
            device=self.model.device
        )
        
        fe = self.model.policy.features_extractor
        fe.autoencoder.to(self.model.device)
        loss = fe.train_autoencoder(obs_batch)  # This should work now
        
        if self.verbose >= 1:
            print(f"AE Loss: {loss:.4f}, Active Dims: {fe.active_dims:.1f}/{fe.autoencoder.max_latent_dim}")

def train_sparse_ae_model(env, total_timesteps=10_000):
    # Create vectorized environment
    from stable_baselines3 import PPO
    import time
    start_time = time.time()
    print("=== Phase 1: Pretraining Sparse Autoencoder ===")
    pretrained_ae, observations = pretrain_ae(env)
     # Load saved data
    save_data = torch.load("pretrained_sparse_ae.pth")
    config = save_data['config']

    # Recreate model architecture
    fe = AdaptiveSparseAE_FeatureExtractor(
        config['observation_space']
        # Include other parameters from config
    )

    # Load weights
    fe.load_state_dict(save_data['state_dict'])
    pretrained_ae = fe.autoencoder
    # After training, prune inactive dimensions
    print("\n=== Pruning Inactive Dimensions ===")
    
    # Get sample of observations for pruning decision
    sample_obs = torch.tensor(np.array(observations[:1000]), dtype=torch.float32)
    inactive_dims = pretrained_ae.prune_inactive_dimensions(sample_obs, threshold=1.0)
    
    # Save pruned autoencoder
    #torch.save(pretrained_ae.state_dict(), "pretrained_sparse_ae.pth")
    print(f"Saved pruned autoencoder with {pretrained_ae.active_dims:.1f} active dimensions")

    # Create model with pretrained autoencoder
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": AdaptiveSparseAE_FeatureExtractor,
            "features_extractor_kwargs": {
                "max_latent_dim": 8,
                "hidden_dims": [128, 64],
                "sparsity_coeff": 0.1,
                "threshold": 1.0,
                "pretrained_ae": pretrained_ae  # Pass the pretrained AE here
            },
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
        },
        verbose=1
    )
    
    # Create and attach callback
    ae_callback = AdaptiveAETrainer(
        train_freq=1000,
        batch_size=512,
        verbose=1
    )
    
    # Learn with callback
    model.learn(
        total_timesteps=total_timesteps,
        callback=ae_callback,
        tb_log_name="sparse_ae_ppo"
    )
    #end_time = time.time()
    #print(f"Training completed in {end_time - start_time:.2f} seconds")
    #time.sleep(100)  # Allow time for logging to complete
    # Plot dimension evolution
    import matplotlib.pyplot as plt
    plt.plot(ae_callback.active_dim_history)
    plt.title("Active Dimensions During Training")
    plt.xlabel("Training Steps (x100)")
    plt.ylabel("Active Dimensions")
    plt.show()
    
    # After training, analyze feature importance
    print("\n=== Analyzing Feature Importance ===")
    
    # Collect sample observations
    sample_obs = []
    obs = env.reset()
    for _ in range(100):
        sample_obs.append(obs[0])
        obs, _, _, _ = env.step([env.action_space.sample() for _ in range(env.num_envs)])
    sample_obs = np.array(sample_obs)

    # Compute and visualize importance
    fe = model.policy.features_extractor
    importance_matrix, active_dims = fe.compute_feature_importance(sample_obs)
    feature_names = [ 
        "vmAllocatedRatio",
        "avgPUUtilization",
        "avgMemoryUtilization",
        "p90MemoryUtiCPUUtilization",
        "p90CPUUtilization",
        "waitingJobsRatioGlobal",
        "waitingJobsRatioRecent"
    ] 
    fe.visualize_feature_importance(importance_matrix, active_dims, feature_names)
    
    
    return model

def pretrain_ae(env, num_samples=10_000, batch_size=512):
    """Pretrain autoencoder with random environment interactions"""
    # Initialize feature extractor
    fe = AdaptiveSparseAE_FeatureExtractor(env.observation_space)
    ae = fe.autoencoder
    optimizer = torch.optim.Adam(ae.parameters())
    
    # Collect random observations
    observations = []
    obs = env.reset()
    for _ in range(num_samples):
        actions = [env.action_space.sample() for _ in range(env.num_envs)]
        obs, _, dones, _ = env.step(actions)
        observations.append(obs[0])  # Use first env's observation
        if any(dones):
            obs = env.reset()
    
    # Convert to tensor dataset
    observations = np.array(observations)
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(observations))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(100):
        epoch_loss = 0
        for batch in loader:
            inputs = batch[0]
            
            # Forward pass
            reconstructed, latent, sparsity_loss = ae(inputs)
            recon_loss = nn.MSELoss()(reconstructed, inputs)
            total_loss = recon_loss + sparsity_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        # Print progress
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.4f} | Active Dims: {ae.active_dims:.1f}/{ae.max_latent_dim}")
        
        # Early stopping if reached target dimensions
        if ae.active_dims <= 5.5:  # Close to target 5
            print(f"Reached target dimensions: {ae.active_dims:.1f}")
            break
    
    # Save pretrained weights
    # torch.save(ae.state_dict(), "pretrained_sparse_ae.pth")
    # print(f"Saved pretrained autoencoder with {ae.active_dims:.1f} active dimensions")
    # Save the FEATURE EXTRACTOR (fe) instead of the autoencoder (ae)
    save_data = {
        'state_dict': fe.state_dict(),  # Save feature extractor's state
        'config': {
            'observation_space': env.observation_space,  # Save observation space shape
            # Include other necessary configs from fe
        }
    }
    torch.save(save_data, "pretrained_sparse_ae.pth")
    print(f"Saved pretrained FEATURE EXTRACTOR with {ae.active_dims:.1f} active dimensions")
    return ae, observations


def compute_feature_importance(autoencoder, observations, n_steps=50):
        # Ensure autoencoder is in eval mode
        autoencoder.eval()
        
        # Convert to tensor
        obs_tensor = torch.tensor(observations, dtype=torch.float32)
        obs_tensor = obs_tensor.squeeze(1)
        # Create baseline (10th percentile of features)
        baseline = torch.quantile(obs_tensor, 0.1, dim=0) * torch.ones_like(obs_tensor)
        
        # Get active dimensions from permanent mask
        active_dims = torch.where(autoencoder.permanent_mask)[0].tolist()
        
        # Initialize importance matrix only for active dimensions
        importance_matrix = np.zeros((len(active_dims), obs_tensor.shape[1]))
        
        # Compute importance only for active dimensions
        for idx, dim in enumerate(active_dims):
            def model_forward(x):
                # Get projected features for specific dimension
                features = autoencoder.encoder(x)
                projected = autoencoder.projection(features)
                return projected[:, dim]
            
            ig = IntegratedGradients(model_forward)
            attr = ig.attribute(obs_tensor, 
                            baselines=baseline,
                            target=None,
                            n_steps=n_steps)
            
            # Average absolute attribution across samples
            importance_matrix[idx] = torch.mean(torch.abs(attr), dim=0).detach().numpy()
        
        return importance_matrix, active_dims