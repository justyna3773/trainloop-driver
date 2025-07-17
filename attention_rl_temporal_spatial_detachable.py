import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from gym import spaces
from captum.attr import IntegratedGradients

class AdaptiveAttentionFeatureExtractor(BaseFeaturesExtractor):
    """Handles 4D input (batch, channel, timesteps, metrics) with adaptive token selection"""
    def __init__(self, observation_space: spaces.Box, 
                 d_embed=64, d_k=32, 
                 min_tokens=2, max_tokens=8,
                 temperature=0.5, reg_strength=0.1):
        assert min_tokens >= 2 and max_tokens <= 8
        features_dim = max_tokens * d_embed
        super().__init__(observation_space, features_dim)
        
        # Token range and regularization
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.temp = temperature
        self.reg_strength = reg_strength
        
        # Process sequential input: (channel, timesteps, metrics)
        self.num_channels = observation_space.shape[0]
        self.num_timesteps = observation_space.shape[1]
        self.num_metrics = observation_space.shape[2]
        
        # Validate input dimensions
        assert self.num_channels == 1, "Expected single channel input"
        
        # Feature embeddings with temporal processing
        self.embedders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.num_timesteps, 32),
                nn.ReLU(),
                nn.Linear(32, d_embed)
            ) for _ in range(self.num_metrics)
        ])
        
        # Token bank (max 8 tokens)
        self.token_bank = nn.Parameter(th.randn(1, max_tokens, d_embed))
        
        # Attention projections
        self.Wq = nn.Linear(d_embed, d_k, bias=False)
        self.Wk = nn.Linear(d_embed, d_k, bias=False)
        self.Wv = nn.Linear(d_embed, d_embed, bias=False)
        
        # Token selector parameters
        self.token_logits = nn.Parameter(th.zeros(max_tokens))
        
        self.d_embed = d_embed
        self.d_k = d_k
        self.reg_loss = th.tensor(0.0)
        self.current_active_tokens = min_tokens
        self.attn_weights = None
        self.selector_weights = None

    def forward(self, x: th.Tensor) -> th.Tensor:
        original_shape = x.shape
        
        # Handle different input cases
        if x.dim() == 2:
            # Flattened input - reshape to original dimensions
            x = x.view(-1, self.num_channels, self.num_timesteps, self.num_metrics)
        elif x.dim() == 4:
            # Proper 4D input (batch, channel, timesteps, metrics)
            pass
        else:
            raise ValueError(f"Unexpected input shape: {original_shape}")
        
        # Remove channel dimension safely
        x = x.squeeze(1)  # Now shape (batch, timesteps, metrics)
        
        # Transpose to (batch, metrics, timesteps)
        x = x.permute(0, 2, 1)  # Now shape (N, 7, 15)
        
        batch_size = x.shape[0]
        
        # 1. Process each metric across timesteps
        embedded_features = []
        for i in range(self.num_metrics):
            # Extract metric i: shape (batch, timesteps) = (N, 15)
            metric_i = x[:, i, :]
            # Process through temporal embedder: (batch, d_embed)
            embedded_i = self.embedders[i](metric_i)
            embedded_features.append(embedded_i)
        
        # Stack features: (batch, num_metrics, d_embed)
        features = th.stack(embedded_features, dim=1)
        
        # 2. Generate token bank
        tokens = self.token_bank.repeat(batch_size, 1, 1)
        
        # 3. Attention between tokens and features
        q = self.Wq(tokens)  # (batch, max_tokens, d_k)
        k = self.Wk(features)  # (batch, num_metrics, d_k)
        v = self.Wv(features)  # (batch, num_metrics, d_embed)
        
        # Attention scores
        scores = th.bmm(q, k.transpose(1, 2)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        self.attn_weights = attn_weights.detach()
        
        # Weighted sum of values
        token_outputs = th.bmm(attn_weights, v)
        
        # 4. Differentiable token selection
        selector_logits = self.token_logits.unsqueeze(0).repeat(batch_size, 1)
        selector_weights = F.gumbel_softmax(selector_logits, tau=self.temp, hard=False, dim=-1)
        self.selector_weights = selector_weights.detach()
        
        # 5. Calculate active tokens
        token_probs = th.sigmoid(self.token_logits.detach())
        self.current_active_tokens = (token_probs > 0.5).sum().item()
        self.current_active_tokens = max(self.min_tokens, min(self.max_tokens, self.current_active_tokens))
        
        # 6. Regularization for token usage
        token_usage = selector_weights.mean(dim=0)
        excess_tokens = F.relu(token_usage.sum() - self.min_tokens)
        self.reg_loss = self.reg_strength * excess_tokens
        
        # 7. Apply selection weights
        selected_outputs = token_outputs * selector_weights.unsqueeze(2)
        return selected_outputs.reshape(batch_size, -1)

    def freeze_weights(self):
        """Freeze all weights in the feature extractor"""
        for param in self.parameters():
            param.requires_grad = False
        print("Feature extractor weights frozen.")


class CustomACPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule, **kwargs):
        # Disable observation flattening
        kwargs['normalize_images'] = False  # Disable image normalization
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        self.reg_loss = th.tensor(0.0)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        # Skip the default preprocessing that flattens observations
        return self.features_extractor(obs)
        
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if hasattr(self.features_extractor, 'reg_loss'):
            self.reg_loss = self.features_extractor.reg_loss
            
        # Use parent class method to get latent features
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        features = self.extract_features(obs)
        if hasattr(self.features_extractor, 'reg_loss'):
            self.reg_loss = self.features_extractor.reg_loss
            
        # Use parent class method to get latent features
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy


class AttentionVisualizationCallback(BaseCallback):
    """Logs attention matrices and IG attributions to TensorBoard"""
    def __init__(self, env, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.env = env
        self.fixed_obs = self._get_fixed_observations()
        self.feature_names = [ "vmAllocatedRatio",
            "avgPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtiCPUUtilization",
            "p90Clization",
            "waitingJobsRatioGlobal",
            "waitingJobsRatioRecent"] # example
        
    def _get_fixed_observations(self, n_samples=5):
        """Create fixed observation samples for consistent attribution"""
        return th.tensor(
            [self.env.observation_space.sample() for _ in range(n_samples)],
            dtype=th.float32
        )
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._log_attention_matrix()
            self._log_ig_attribution()
            self._log_active_tokens()
        return True
    

    def _log_attention_matrix(self):
        """Log feature-to-query attention matrix - show only active tokens"""
        feat_extractor = self.model.policy.features_extractor
        if feat_extractor.attn_weights is None:
            return
            
        # Get active tokens count and attention weights
        active_tokens = feat_extractor.current_active_tokens
        attn = feat_extractor.attn_weights
        avg_attn = attn.mean(dim=0).cpu().numpy()  # [max_tokens, 7]
        
        # Filter to active tokens only
        active_attn = avg_attn[:active_tokens, :]  # [active_tokens, 7]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(active_attn, cmap='viridis')
        
        # Set labels
        ax.set_title(f'Feature-to-Query Attention (Active Tokens: {active_tokens})')
        ax.set_xlabel('Input Features')
        ax.set_ylabel('Active Query Tokens')
        fig.colorbar(cax, label='Attention Weight')
        
        # Set ticks - only active tokens on y-axis
        ax.set_xticks(np.arange(7))
        ax.set_yticks(np.arange(active_tokens))
        ax.set_xticklabels(self.feature_names)
        ax.set_yticklabels([f'Token {i}' for i in range(active_tokens)])
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        
        # Log to TensorBoard
        self.logger.record(
            "attention/feature_to_query", 
            Figure(fig, close=True), 
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig)
    
    def _log_ig_attribution(self):
        """Compute and log IG feature attributions"""
        feat_extractor = self.model.policy.features_extractor
        
        # Temporarily switch to eval mode
        was_training = feat_extractor.training
        feat_extractor.eval()
        
        try:
            # Create scalar wrapper for feature extractor
            def scalar_forward(x):
                outputs = feat_extractor(x)
                return outputs.norm(dim=1)  # Convert to scalar
                
            # Create IG instance
            ig = IntegratedGradients(scalar_forward)
            
            # Compute attributions for fixed samples
            all_attributions = []
            for obs in self.fixed_obs:
                obs = obs.unsqueeze(0).to(self.model.device)
                baseline = th.zeros_like(obs)
                
                # Compute attributions
                attr = ig.attribute(obs, baselines=baseline, n_steps=25)
                all_attributions.append(attr.detach().cpu())
            
            # Average attributions
            avg_attr = th.cat(all_attributions).mean(dim=0).squeeze()
            
            # Handle different attribution dimensions
            if avg_attr.dim() == 3:  # (batch, timesteps, metrics)
                # Average across timesteps to get per-metric attribution
                avg_attr = avg_attr.mean(dim=1)
            elif avg_attr.dim() == 2:  # (timesteps, metrics)
                # Average across timesteps
                avg_attr = avg_attr.mean(dim=0)
            elif avg_attr.dim() > 1:
                # Flatten to 1D if needed
                avg_attr = avg_attr.mean(dim=0)
            
            # Convert to numpy and ensure 1D array
            attr_np = avg_attr.numpy().flatten()
            
            # Ensure we have the correct number of features
            num_features = len(self.feature_names)
            if len(attr_np) != num_features:
                # If we have too many values, take the first num_features
                attr_np = attr_np[:num_features]
            
            # Create attribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['skyblue' if a > 0 else 'salmon' for a in attr_np]
            ax.bar(range(len(attr_np)), attr_np, color=colors)
            ax.set_title("Feature Attribution via Integrated Gradients")
            ax.set_xlabel("Input Feature")
            ax.set_ylabel("Attribution Score")
            
            # Set ticks and labels
            ax.set_xticks(range(len(attr_np)))
            ax.set_xticklabels(self.feature_names, rotation=90, ha='center')
            
            # Log to TensorBoard
            self.logger.record(
                "attention/ig_attribution", 
                Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close(fig)
            
        finally:
            # Restore original training mode
            feat_extractor.train(was_training)
    
    def _log_active_tokens(self):
        """Log current active token count"""
        feat_extractor = self.model.policy.features_extractor
        active_tokens = feat_extractor.current_active_tokens
        print(f"Step {self.num_timesteps}: Active tokens = {active_tokens}")
        self.logger.record("attention/active_tokens", active_tokens)


def train_model(env, tensorboard_log='.', total_timesteps=10_000, callback=None, 
               freeze_feature_extractor=True, policy_kwargs=None):
    from stable_baselines3 import PPO
    
    # Use custom policy_kwargs if provided, else create default
    if policy_kwargs is None:
        policy_kwargs = {
            "features_extractor_class": AdaptiveAttentionFeatureExtractor,
            "features_extractor_kwargs": {
                "d_embed": 64,
                "d_k": 32,
                "min_tokens": 2,
                "max_tokens": 8,
                "temperature": 0.5,
                "reg_strength": 0.1
            },
            "net_arch": [{"pi": [64, 64], "vf": [64, 64]}]
        }

    # Create PPO model with custom policy
    model = PPO(
        CustomACPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./attention_exp_temporal_spatial",
    )
    
    # Create visualization callback
    vis_callback = AttentionVisualizationCallback(env, log_freq=1000)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[vis_callback, callback] if callback else [vis_callback],
        tb_log_name="attention_exp_temporal_spatial"
    )
    # Freeze feature extractor if requested
    if freeze_feature_extractor:
        model.policy.features_extractor.freeze_weights()
    model.save("pretrained_attention_model")
     # Define custom objects for loading
    custom_objects = {
        "policy_class": CustomACPolicy,
        "features_extractor_class": AdaptiveAttentionFeatureExtractor
    }
    
    # Load the pretrained model
    loaded_model = PPO.load(
        "pretrained_attention_model",
        env=env,
        custom_objects=custom_objects
    )
    
    # Freeze the feature extractor
    if freeze_feature_extractor:
        loaded_model.policy.features_extractor.freeze_weights()
    
    return loaded_model








#TODO IG part

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from stable_baselines3 import PPO


def load_model(model_path="pretrained_attention_model"):
    # Define custom objects for loading
    custom_objects = {
        "policy_class": CustomACPolicy,
        "features_extractor_class": AdaptiveAttentionFeatureExtractor
    }
    
    # Load the pretrained model
    model = PPO.load(
        model_path,
        custom_objects=custom_objects
    )
    
    print("Model loaded successfully")
    return model

def run_ig_analysis(model, observations, feature_names=None, baseline_type="zero", n_steps=25):
    """
    Perform Integrated Gradients attribution analysis using precomputed observations
    
    Args:
        model: Loaded RL model
        observations: NumPy array of observations (shape: [n_samples, ...])
        feature_names: List of feature names for visualization
        baseline_type: Type of baseline ("zero" or "mean")
        n_steps: Number of steps for IG approximation
    """
    # Get feature extractor and device
    feat_extractor = model.policy.features_extractor
    device = model.device
    feat_extractor.eval()  # Switch to evaluation mode
    
    # Convert observations to tensor
    obs_tensor = th.tensor(observations, dtype=th.float32).to(device)
    
    # Create baseline
    if baseline_type == "zero":
        baseline = th.zeros_like(obs_tensor)
    elif baseline_type == "mean":
        baseline = th.mean(obs_tensor, dim=0, keepdim=True).repeat(obs_tensor.shape[0], 1)
    else:
        raise ValueError(f"Invalid baseline type: {baseline_type}")
    
    # Define scalar wrapper for feature extractor
    def scalar_forward(x):
        outputs = feat_extractor(x)
        return outputs.norm(dim=1)  # Convert to scalar
    
    # Create IG instance
    ig = IntegratedGradients(scalar_forward)
    
    # Compute attributions in batches
    batch_size = 8  # Adjust based on available memory
    all_attributions = []
    
    for i in range(0, len(obs_tensor), batch_size):
        batch_obs = obs_tensor[i:i+batch_size]
        batch_baseline = baseline[i:i+batch_size]
        
        # Compute attributions
        attr = ig.attribute(batch_obs, 
                            baselines=batch_baseline, 
                            n_steps=n_steps,
                            internal_batch_size=4)  # Reduce memory usage
        
        all_attributions.append(attr.detach().cpu())
    
    # Combine results
    attributions = th.cat(all_attributions)
    
    # Process and visualize results
    results = []
    for i, attr in enumerate(attributions):
        # Determine original observation shape
        if hasattr(model.observation_space, "shape"):
            obs_shape = model.observation_space.shape
        else:
            # Infer from data if observation_space not available
            obs_shape = observations[0].shape
        
        # Reshape attribution to match observation structure
        attr = attr.reshape(obs_shape)
        
        # Process different dimensionalities
        if len(obs_shape) == 3:  # (channels, timesteps, metrics)
            # Average across channels and timesteps
            attr_reduced = attr.mean(dim=0).mean(dim=0)
        elif len(obs_shape) == 2:  # (timesteps, metrics)
            # Average across timesteps
            attr_reduced = attr.mean(dim=0)
        elif len(obs_shape) == 1:  # Flattened vector
            attr_reduced = attr
        else:
            # Handle higher dimensions by flattening
            attr_reduced = attr.flatten()
        
        # Convert to numpy
        attr_np = attr_reduced.numpy()
        num_features = len(attr_np)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['skyblue' if a > 0 else 'salmon' for a in attr_np]
        ax.bar(range(num_features), attr_np, color=colors)
        ax.set_title(f"Feature Attribution via IG (Sample {i+1})")
        ax.set_xlabel("Input Feature")
        ax.set_ylabel("Attribution Score")
        
        # Set feature names if available
        if feature_names:
            if len(feature_names) > num_features:
                feature_names = feature_names[:num_features]
            elif len(feature_names) < num_features:
                feature_names += [f"Feature {j}" for j in range(len(feature_names), num_features)]
            
            ax.set_xticks(range(num_features))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
        else:
            ax.set_xticks(range(num_features))
            ax.set_xticklabels([f"Feat {j}" for j in range(num_features)], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"ig_attribution_sample_{i+1}.png")
        plt.show()
        #plt.close(fig)
        
        # Save results
        results.append({
            "sample_index": i,
            "attribution": attr_np,
            "feature_names": feature_names[:num_features] if feature_names else None,
            "plot_file": f"ig_attribution_sample_{i+1}.png"
        })
    
    return results, fig

# Example usage:
if __name__ == "__main__":
    # Load your model first (using the load_model function from previous implementation)
    # model = load_model()
    
    # Generate or load your observations
    # observations = np.load("observations.npy")  # Shape: [n_samples, ...]
    
    # Define feature names (if available)
    feature_names = [
        "vmAllocatedRatio",
        "avgPUUtilization",
        "avgMemoryUtilization",
        "p90MemoryUtiCPUUtilization",
        "p90Clization",
        "waitingJobsRatioGlobal",
        "waitingJobsRatioRecent"
    ]
    
    # Run IG analysis
    results = run_ig_analysis(
        model=model,
        observations=observations,
        feature_names=feature_names,
        baseline_type="mean",  # Use mean observation as baseline
        n_steps=50  # More steps for better accuracy
    )
    
    print(f"IG analysis completed for {len(results)} samples")