import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import os


class DynamicFeatureReducer(nn.Module):
    def __init__(self, input_dim=7, max_reduced_dim=4, min_reduced_dim=1, 
                 key_dim=8, value_dim=8, temp_init=1.0, temp_decay=0.99):
        super().__init__()
        self.input_dim = input_dim
        self.max_reduced_dim = max_reduced_dim
        self.min_reduced_dim = min_reduced_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.temp = temp_init
        self.temp_decay = temp_decay
        
        # Learnable reduction factor (sigmoid ensures 0-1 range)
        self.reduction_factor = nn.Parameter(th.tensor([0.5]))
        
        # Feature selection parameters
        self.selector_logits = nn.Parameter(th.randn(input_dim))
        
        # Attention mechanism components
        self.key_layer = nn.Linear(1, key_dim)
        self.value_layer = nn.Linear(1, value_dim)
        self.queries = nn.Parameter(th.randn(max_reduced_dim, key_dim))
        self.projection = nn.Linear(value_dim, 1)
        
        # Store attention weights for visualization
        self.attention_weights = None
        self.selection_mask = None

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Calculate current reduced dimension
        reduced_dim = self.get_current_reduced_dim()
        
        # Differentiable feature selection using Gumbel-Softmax
        if self.training:
            # Use Gumbel-Softmax for differentiable sampling
            selector = F.gumbel_softmax(
                self.selector_logits.repeat(batch_size, 1), 
                tau=self.temp, 
                hard=True
            )
        else:
            # During evaluation, use deterministic threshold
            selection_probs = th.sigmoid(self.selector_logits)
            selector = (selection_probs > 0.5).float().unsqueeze(0).repeat(batch_size, 1)
        
        self.selection_mask = selector  # Store for visualization
        x_selected = x * selector
        
        # Process selected features through attention
        x_reshaped = x_selected.view(-1, 1)
        keys = self.key_layer(x_reshaped).view(batch_size, self.input_dim, -1)
        values = self.value_layer(x_reshaped).view(batch_size, self.input_dim, -1)
        
        # Use only the required number of queries
        active_queries = self.queries[:reduced_dim]
        
        scores = th.matmul(keys, active_queries.T.unsqueeze(0))
        scores = scores.permute(0, 2, 1)
        self.attention_weights = F.softmax(scores, dim=-1)
        
        output = th.matmul(self.attention_weights, values)
        output = self.projection(output).squeeze(-1)
        
        # Pad output to max_reduced_dim for consistent tensor shape
        if reduced_dim < self.max_reduced_dim:
            pad_size = self.max_reduced_dim - reduced_dim
            padding = th.zeros((batch_size, pad_size), device=output.device)
            output = th.cat([output, padding], dim=1)
            
        return output

    def get_current_reduced_dim(self):
        """Calculate current reduced dimension based on learned factor"""
        # Sigmoid ensures value between 0 and 1
        factor = th.sigmoid(self.reduction_factor).item()
        
        # Scale to desired range
        reduced_dim = int(self.min_reduced_dim + 
                         factor * (self.max_reduced_dim - self.min_reduced_dim))
        
        # Ensure within bounds
        return max(self.min_reduced_dim, min(reduced_dim, self.max_reduced_dim))

    def update_temperature(self):
        """Anneals the temperature for Gumbel-Softmax"""
        self.temp *= self.temp_decay
        
    def get_feature_importance(self):
        """Get feature selection probabilities"""
        with th.no_grad():
            return th.sigmoid(self.selector_logits).cpu().numpy()
    # New function in DynamicFeatureReducer
    def freeze_adaptation(self):
        """Lock adaptive components for attribution methods"""
        # Use deterministic feature selection
        selection_probs = th.sigmoid(self.selector_logits)
        self.fixed_selector = (selection_probs > 0.5).float()
        
        # Use maximum reduced dimension
        self.fixed_reduced_dim = self.max_reduced_dim
        self.active_queries = self.queries[:self.max_reduced_dim]
        
        # Disable temperature updates
        self.training = False




class AdaptiveAttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, max_reduced_dim=4, min_reduced_dim=1, 
                 key_dim=8, value_dim=8, temp_init=1.0, temp_decay=0.99):
        # Features_dim is set to max_reduced_dim as the output dimension
        super().__init__(observation_space, features_dim=max_reduced_dim)
        self.input_dim = observation_space.shape[0]
        
        self.reducer = DynamicFeatureReducer(
            input_dim=self.input_dim,
            max_reduced_dim=max_reduced_dim,
            min_reduced_dim=min_reduced_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            temp_init=temp_init,
            temp_decay=temp_decay
        )
    
    def forward(self, observations):
        return self.reducer(observations)
    
    def update_temperature(self):
        self.reducer.update_temperature()
        
    def get_current_reduced_dim(self):
        return self.reducer.get_current_reduced_dim()
    
    def get_feature_importance(self):
        return self.reducer.get_feature_importance()
    # Add to AdaptiveAttentionFeatureExtractor
    def freeze_adaptation(self):
        self.reducer.freeze_adaptation()

class TemperatureUpdateCallback(BaseCallback):
    """
    Callback to update the temperature parameter of the feature selector
    during training. Should be called periodically (e.g., after each rollout)
    """
    def __init__(self, update_freq=1000, verbose=0):
        super().__init__(verbose)
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            # Update temperature in the feature extractor
            if hasattr(self.model.policy, "features_extractor"):
                self.model.policy.features_extractor.update_temperature()
                
                # Log current reduced dimension
                curr_dim = self.model.policy.features_extractor.get_current_reduced_dim()
                self.logger.record("train/reduced_dim", curr_dim)
                
                # Log current temperature
                temp = self.model.policy.features_extractor.reducer.temp
                self.logger.record("train/selector_temp", temp)
        return True

def train_model(env):
    from stable_baselines3 import PPO
    # Policy configuration with adaptive feature extractor
    policy_kwargs = dict(
        features_extractor_class=AdaptiveAttentionFeatureExtractor,
        features_extractor_kwargs=dict(
            max_reduced_dim=8,          # Maximum reduced dimension
            min_reduced_dim=2,           # Minimum reduced dimension
            key_dim=16,                  # Attention key dimension
            value_dim=16,                # Attention value dimension
            temp_init=1.0,               # Initial temperature
            temp_decay=0.995             # Temperature decay rate
        ),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Policy network architecture
    )

    # Create model with callback
    model = PPO(
        "MlpPolicy", 
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto"
    )

    # Create temperature update callback
    #temp_callback = TemperatureUpdateCallback(update_freq=1000)  # Update every 1000 steps
    
    # Create callback instance
    viz_callback = AttentionVisualizationCallback(
        log_interval=1000  # Log every 1000 steps
    )

    # Combine with your temperature callback
    callbacks = [
        TemperatureUpdateCallback(update_freq=1000, log_interval=1),  # Update temperature every 1000 steps
        viz_callback, 

    ]

    # Train the model
    model.learn(
        total_timesteps=10_000,
        callback=callbacks,
    
    )

    # After training, you can inspect the final configuration
    final_dim = model.policy.features_extractor.get_current_reduced_dim()
    feature_importance = model.policy.features_extractor.get_feature_importance()

    print(f"Final reduced dimension: {final_dim}")
    print("Feature importance scores:")
    for i, score in enumerate(feature_importance):
        print(f"Feature {i}: {score:.4f}")
    return model

#TODO this is for TensorBoard visualization, but it doesnt get saved
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class AttentionVisualizationCallback(BaseCallback):
    """
    Callback to visualize attention weights and feature importance during training
    without interrupting the learning process.
    """
    def __init__(self, log_interval=1000, metric_names=None, verbose=0):
        """
        Args:
            log_interval: Log every N steps
            metric_names: List of names for your cloud metrics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        self.metric_names = metric_names or [
            f"Metric_{i}" for i in range(7)  # Default names if not provided
        ]
        self.log_dir='./'
        # Initialize storage
        self.attention_history = []
        self.importance_history = []
        self.step_history = []
        self.logger.set_dir(os.getcwd())

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            # Get feature extractor
            fe = self.model.policy.features_extractor
            
            # 1. Get feature importance
            feature_importance = fe.get_feature_importance()
            self.importance_history.append(feature_importance)
            
            # 2. Get attention weights if available
            if fe.reducer.attention_weights is not None:
                # Process attention weights
                attn = fe.reducer.attention_weights
                
                # Detach, convert to numpy, and average over batch
                attn_np = attn.detach().cpu().numpy()
                mean_attn = attn_np.mean(axis=0)  # Average across batch dimension
                
                # Store only active queries (non-zero rows)
                current_dim = fe.get_current_reduced_dim()
                active_attn = mean_attn[:current_dim, :]
                self.attention_history.append(active_attn)
                self.step_history.append(self.num_timesteps)
                
                # Create visualizations
                self._log_attention_heatmap(active_attn)
                self._log_feature_importance(feature_importance)
                
        return True

    def _log_attention_heatmap(self, attention_weights):
        """Create and log attention heatmap"""
        fig, ax = plt.subplots(figsize=(10, 6))
        log_dir = self.logger.get_dir()
        print(f"Logging attention heatmap to {log_dir}")
        # Create heatmap
        cax = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        fig.colorbar(cax, label='Attention Weight')
        
        # Set labels
        ax.set_xlabel('Input Features')
        ax.set_ylabel('Queries')
        ax.set_title(f'Attention Patterns (Step {self.num_timesteps})')
        
        # Set tick labels
        ax.set_xticks(np.arange(len(self.metric_names)))
        ax.set_xticklabels(self.metric_names, rotation=45, ha='right')
        ax.set_yticks(np.arange(attention_weights.shape[0]))
        ax.set_yticklabels([f'Q{i}' for i in range(attention_weights.shape[0])])
        
        # Log to TensorBoard
        self.logger.record(
            os.path.join(self.log_dir, "attention/heatmap"),
            Figure(fig, close=True),  # Automatically close figure after logging
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig)

    def _log_feature_importance(self, importance):
        """Create and log feature importance bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        colors = plt.cm.viridis(importance / np.max(importance))
        bars = ax.bar(range(len(importance)), importance, color=colors)
        
        # Add values on top
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Set labels
        ax.set_xlabel('Features')
        ax.set_ylabel('Selection Probability')
        ax.set_title(f'Feature Importance (Step {self.num_timesteps})')
        ax.set_xticks(range(len(self.metric_names)))
        ax.set_xticklabels(self.metric_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)  # Importance probabilities are between 0-1
        
        # Log to TensorBoard
        self.logger.record(
            os.path.join(self.log_dir,"attention/importance"),
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig)

    def get_final_insights(self):
        """Return final attention patterns and importance scores after training"""
        return {
            "attention_weights": self.attention_history[-1],
            "feature_importance": self.importance_history[-1],
            "step_history": self.step_history
        }