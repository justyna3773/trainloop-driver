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
    """Learns optimal token count (2-8) with Gumbel-Softmax selection"""
    def __init__(self, observation_space: spaces.Box, 
                 d_embed=64, d_k=32, 
                 min_tokens=2, max_tokens=8,
                 temperature=0.5, reg_strength=0.1,
                 freeze=False):  # ADDED freeze parameter
        assert min_tokens >= 2 and max_tokens <= 8
        features_dim = max_tokens * d_embed
        super().__init__(observation_space, features_dim)
        
        # Token range and regularization
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.temp = temperature
        self.reg_strength = reg_strength
        self.freeze = freeze  # ADDED freeze state
        
        # Feature embeddings
        self.embedders = nn.ModuleList([nn.Linear(1, d_embed) for _ in range(7)])
        
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

        # ADDED: Freeze weights if requested
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
                
    # ADDED: Method to load pretrained weights
    def load_pretrained(self, state_dict):
        """Load pretrained weights and freeze if needed"""
        self.load_state_dict(state_dict)
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_size = x.shape[0]
        
        # 1. Feature Embedding
        embedded_features = []
        for i in range(7):
            feat_i = x[:, i].unsqueeze(1)
            embedded_i = self.embedders[i](feat_i)
            embedded_features.append(embedded_i)
        features = th.stack(embedded_features, dim=1)
        
        # 2. Generate All Tokens
        tokens = self.token_bank.repeat(batch_size, 1, 1)
        
        # 3. Attention Processing
        q = self.Wq(tokens)
        k = self.Wk(features)
        v = self.Wv(features)
        
        scores = th.bmm(q, k.transpose(1, 2)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        self.attn_weights = attn_weights.detach()
        token_outputs = th.bmm(attn_weights, v)
        
        # 4. Differentiable Token Selection
        selector_logits = self.token_logits.unsqueeze(0).repeat(batch_size, 1)
        selector_weights = F.gumbel_softmax(selector_logits, tau=self.temp, hard=False, dim=-1)
        self.selector_weights = selector_weights.detach()
        
        # 5. Calculate active tokens
        token_probs = th.sigmoid(self.token_logits.detach())
        self.current_active_tokens = (token_probs > 0.5).sum().item()
        self.current_active_tokens = max(self.min_tokens, min(self.max_tokens, self.current_active_tokens))
        
        # 6. Calculate Regularization
        token_usage = selector_weights.mean(dim=0)
        excess_tokens = F.relu(token_usage.sum() - self.min_tokens)
        self.reg_loss = self.reg_strength * excess_tokens
        
        # 7. Apply selection weights
        selected_outputs = token_outputs * selector_weights.unsqueeze(2)
        return selected_outputs.reshape(batch_size, -1)


# Policy for standard PPO (MlpPolicy)
class CustomACPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_loss = th.tensor(0.0)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if hasattr(self.features_extractor, 'reg_loss'):
            self.reg_loss = self.features_extractor.reg_loss
        return super().forward(features, deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        values, log_prob, entropy = super().evaluate_actions(obs, actions)
        if hasattr(self, 'reg_loss'):
            reg_loss = self.reg_loss.to(values.device)
            return values, log_prob, entropy - reg_loss
        return values, log_prob, entropy


# Policy for RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

class CustomRecurrentACPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_loss = th.tensor(0.0)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        features = super().extract_features(obs)
        if hasattr(self.features_extractor, 'reg_loss'):
            self.reg_loss = self.features_extractor.reg_loss
        return features

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False
    ) -> tuple:
        features = self.extract_features(obs)
        latent_pi, lstm_states_pi = self._process_sequence(
            features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states.vf

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return (
            actions, 
            values, 
            log_prob, 
            RNNStates(lstm_states_pi, lstm_states_vf)
        )

    def evaluate_actions(
        self, 
        obs: th.Tensor, 
        actions: th.Tensor, 
        lstm_states: RNNStates, 
        episode_starts: th.Tensor
    ) -> tuple:
        features = self.extract_features(obs)
        latent_pi, lstm_states_pi = self._process_sequence(
            features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states.vf

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        if hasattr(self, 'reg_loss'):
            reg_loss = self.reg_loss.to(values.device)
            entropy = entropy - reg_loss
        
        return (
            values, 
            log_prob, 
            entropy, 
            RNNStates(lstm_states_pi, lstm_states_vf)
        )

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
            "waitingJobsRatioRecent"]
        
    def _get_fixed_observations(self, n_samples=5):
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
        feat_extractor = self.model.policy.features_extractor
        if feat_extractor.attn_weights is None:
            return
            
        active_tokens = feat_extractor.current_active_tokens
        attn = feat_extractor.attn_weights
        avg_attn = attn.mean(dim=0).cpu().numpy()
        active_attn = avg_attn[:active_tokens, :]

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(active_attn, cmap='viridis')
        ax.set_title(f'Feature-to-Query Attention (Active Tokens: {active_tokens})')
        ax.set_xlabel('Input Features')
        ax.set_ylabel('Active Query Tokens')
        fig.colorbar(cax, label='Attention Weight')
        ax.set_xticks(np.arange(7))
        ax.set_yticks(np.arange(active_tokens))
        ax.set_xticklabels(self.feature_names)
        ax.set_yticklabels([f'Token {i}' for i in range(active_tokens)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        
        self.logger.record(
            "attention/feature_to_query", 
            Figure(fig, close=True), 
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig)
    
    def _log_ig_attribution(self):
        feat_extractor = self.model.policy.features_extractor
        was_training = feat_extractor.training
        feat_extractor.eval()
        
        try:
            def scalar_forward(x):
                return feat_extractor(x).norm(dim=1)
                
            ig = IntegratedGradients(scalar_forward)
            all_attributions = []
            for obs in self.fixed_obs:
                obs = obs.unsqueeze(0).to(self.model.device)
                baseline = th.zeros_like(obs)
                attr = ig.attribute(obs, baselines=baseline, n_steps=25)
                all_attributions.append(attr.detach().cpu())
            
            avg_attr = th.cat(all_attributions).mean(dim=0).squeeze()
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['skyblue' if a > 0 else 'salmon' for a in avg_attr.numpy()]
            ax.bar(range(7), avg_attr.numpy(), color=colors)
            ax.set_title("Feature Attribution via Integrated Gradients")
            ax.set_xlabel("Input Feature")
            ax.set_ylabel("Attribution Score")
            ax.set_xticks(range(7))
            ax.set_xticklabels(self.feature_names, rotation=90, ha='center')
            
            self.logger.record(
                "attention/ig_attribution", 
                Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close(fig)
            
        finally:
            feat_extractor.train(was_training)
    
    def _log_active_tokens(self):
        feat_extractor = self.model.policy.features_extractor
        active_tokens = feat_extractor.current_active_tokens
        self.logger.record("attention/active_tokens", active_tokens)

def train_model(env, policy_type='RecurrentPPO', total_timesteps=10_000, 
                tensorboard_log='.', callback=None,
                freeze_after_pretrain=True):  # NEW: Add freeze control
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    
   
    model, tb_log_name = prepare_model(env, policy_type=policy_type, tensorboard_log=tensorboard_log, pretrained_feature_extractor=None)

    vis_callback = AttentionVisualizationCallback(env, log_freq=1000)
    callbacks = [vis_callback]
    if callback:
        callbacks.append(callback)
    
    # pretrain for total_timesteps
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name
    )
    # Freeze feature extractor
    #callback.save_to_csv(filename=f'../models/Attention_{policy_type}_pretraining.csv')  # Save training data to CSV
    pretrained_feature_extractor = model.policy.features_extractor.state_dict()

    model, tb_log_name = prepare_model(env, policy_type=policy_type, tensorboard_log=tensorboard_log, pretrained_feature_extractor=pretrained_feature_extractor)
    
    model.policy.features_extractor.load_pretrained(pretrained_feature_extractor)
    
    # Return model and feature extractor state
    return model, model.policy.features_extractor.state_dict()

def prepare_model(env, policy_type='RecurrentPPO', pretrained_feature_extractor=None, tensorboard_log='.'):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    
    freeze = pretrained_feature_extractor is not None
    policy_kwargs = {
        "features_extractor_class": AdaptiveAttentionFeatureExtractor,
        "features_extractor_kwargs": {
            "d_embed": 64,
            "d_k": 32,
            "min_tokens": 2,
            "max_tokens": 8,
            "temperature": 0.5,
            "reg_strength": 0.1,
            "freeze": freeze  # Now respects user choice
        },
    }
    
    # ===== Model setup =====
    if policy_type == 'RecurrentPPO':
        policy_kwargs.update({
            "net_arch": [{"pi": [64, 64], "vf": [64, 64]}],
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "enable_critic_lstm": True
        })
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log
        )
        tb_log_name = "attention_exp_recurrent"
    else:  # MlpPolicy
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log
        )
        tb_log_name = "attention_exp_mlp"
    return model, tb_log_name
# def train_model(env, policy_type='RecurrentPPO', total_timesteps=10_000, 
#                 tensorboard_log='.', callback=None,
#                 pretrained_feature_extractor=None):  # ADDED pretrained parameter
#     from stable_baselines3 import PPO
#     from sb3_contrib import RecurrentPPO
    
#     # ADDED freeze option based on pretrained availability
    
#     freeze = pretrained_feature_extractor is not None
    
#     policy_kwargs = {
#         "features_extractor_class": AdaptiveAttentionFeatureExtractor,
#         "features_extractor_kwargs": {
#             "d_embed": 64,
#             "d_k": 32,
#             "min_tokens": 2,
#             "max_tokens": 8,
#             "temperature": 0.5,
#             "reg_strength": 0.1,
#             "freeze": freeze  # ADDED freeze parameter
#         },
#     }
    
#     if policy_type == 'RecurrentPPO':
#         policy_kwargs.update({
#             "net_arch": [{"pi": [64, 64], "vf": [64, 64]}],
#             "lstm_hidden_size": 256,
#             "n_lstm_layers": 1,
#             "enable_critic_lstm": True
#         })
#         model = RecurrentPPO(
#             "MlpLstmPolicy",
#             env,
#             policy_kwargs=policy_kwargs,
#             verbose=1,
#             tensorboard_log=tensorboard_log
#         )
#         tb_log_name = "attention_exp_recurrent"
#     elif policy_type == 'MlpPolicy':
#         model = PPO(
#             "MlpPolicy",
#             env,
#             policy_kwargs=policy_kwargs,
#             verbose=1,
#             tensorboard_log=tensorboard_log
#         )
#         tb_log_name = "attention_exp_mlp"
    
#     # ADDED: Load pretrained weights if provided
#     if pretrained_feature_extractor is not None:
#         model.policy.features_extractor.load_pretrained(pretrained_feature_extractor)
#     else:
#         #pretrain_model before freezing weights
#         model.learn()
#         vis_callback = AttentionVisualizationCallback(env, log_freq=1000)
    
#         callbacks = [vis_callback]
#         if callback:
#             callbacks.append(callback)
        
#         model.learn(
#             total_timesteps=total_timesteps,
#             callback=callbacks,
#             tb_log_name=tb_log_name
#         )

#         # Extract pretrained weights
#         pretrained_weights = model.policy.features_extractor.state_dict()
#         # freeze weights after pretraining

#     return model, pretrained_weights

# def pretrain_feature_extractor(env):
#     # Step 1: Pretrain the feature extractor
#     pretrain_model = train_model(
#         env,
#         policy_type='MlpPolicy',
#         total_timesteps=5_000,
#         tensorboard_log='./pretrain_logs'
#     )

#     # Extract pretrained weights
#     pretrained_weights = pretrain_model.policy.features_extractor.state_dict()

#     # Step 2: Train with frozen features
#     final_model = train_model(
#         env,
#         policy_type='MlpPolicy',
#         total_timesteps=50000,
#         tensorboard_log='./main_logs',
#         pretrained_feature_extractor=pretrained_weights
#     )

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from captum.attr import IntegratedGradients
from gym import Env, spaces
import numpy as np

def create_dummy_env(observation_shape, action_space_type="Discrete", n_actions=7):
    """
    Create a minimal environment for model loading
    
    Args:
        observation_shape: Tuple defining observation shape (e.g., (7,))
        action_space_type: "Discrete" or "Box"
        n_actions: For Discrete spaces, number of actions
    
    Returns:
        A dummy environment with specified properties
    """
    class DummyEnv(Env):
        def __init__(self):
            self.observation_space = spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=observation_shape,
                dtype=np.float32
            )
            
            if action_space_type == "Discrete":
                self.action_space = spaces.Discrete(n_actions)
            else:
                self.action_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n_actions,),
                    dtype=np.float32
                )
            
        def reset(self):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        def step(self, action):
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                False,
                {}
            )
    
    return DummyEnv()

def load_model(model_path, env=None, policy_type='RecurrentPPO', 
              obs_shape=(7,), n_actions=7, action_space_type="Discrete"):
    """
    Load model with automatic dummy environment creation
    
    Args:
        model_path: Path to saved model
        env: Existing environment (optional)
        policy_type: 'RecurrentPPO' or 'MlpPolicy'
        obs_shape: Observation shape for dummy env
        n_actions: Number of actions for dummy env
        action_space_type: "Discrete" or "Box"
    """
    # Define custom objects
    custom_objects = {
        "policy_class": CustomRecurrentACPolicy if policy_type == 'RecurrentPPO' else CustomACPolicy,
        "features_extractor_class": AdaptiveAttentionFeatureExtractor
    }
    
    # Create dummy environment if needed
    if policy_type == 'RecurrentPPO' and env is None:
        print("Creating dummy environment for RecurrentPPO loading")
        env = create_dummy_env(
            observation_shape=obs_shape,
            action_space_type=action_space_type,
            n_actions=n_actions
        )
    
    # Load the model
    if policy_type == 'RecurrentPPO':
        model = RecurrentPPO.load(
            model_path,
            env=env,
            custom_objects=custom_objects
        )
    else:
        model = PPO.load(
            model_path,
            custom_objects=custom_objects
        )
    
    print(f"Loaded {policy_type} model from {model_path}")
    return model

# def load_model(model_path, env=None, policy_type='PPO'):
#     """
#     Load a pretrained model with custom feature extractor and policy
    
#     Args:
#         model_path: Path to the saved model
#         env: Environment (optional, needed for RecurrentPPO)
#         policy_type: Type of policy ('RecurrentPPO' or 'MlpPolicy')
    
#     Returns:
#         Loaded model ready for inference or analysis
#     """
#     # Define custom objects
#     custom_objects = {
#         "policy_class": CustomRecurrentACPolicy if policy_type == 'RecurrentPPO' else CustomACPolicy,
#         "features_extractor_class": AdaptiveAttentionFeatureExtractor
#     }
    
#     # Load the model
#     if policy_type == 'RecurrentPPO':
#         if env is None:
#             raise ValueError("Environment must be provided for RecurrentPPO models")
#         model = RecurrentPPO.load(
#             model_path,
#             env=env,
#             custom_objects=custom_objects
#         )
#     else:
#         model = PPO.load(
#             model_path,
#             custom_objects=custom_objects
#         )
    
#     print(f"Loaded {policy_type} model from {model_path}")
#     return model

def compute_attention_matrix(model, observations, feature_names=None):
    """
    Compute and visualize the attention matrix for given observations
    
    Args:
        model: Loaded RL model
        observations: NumPy array of observations (shape: [n_samples, 7])
        feature_names: List of feature names for visualization
    
    Returns:
        attention_matrix: Computed attention weights (n_samples, active_tokens, n_features)
        fig: Matplotlib figure object
    """
    # Get feature extractor and device
    feat_extractor = model.policy.features_extractor
    device = model.device
    
    # Convert observations to tensor
    obs_tensor = th.tensor(observations, dtype=th.float32).to(device)
    
    # Run through feature extractor to capture attention
    with th.no_grad():
        _ = feat_extractor(obs_tensor)
    
    # Get attention weights and active token count
    attn_weights = feat_extractor.attn_weights.cpu().numpy()
    active_tokens = feat_extractor.current_active_tokens
    
    # Average across batch if multiple samples
    if len(observations) > 1:
        avg_attn = attn_weights.mean(axis=0)
    else:
        avg_attn = attn_weights[0]
    
    # Filter to active tokens only
    active_attn = avg_attn[:active_tokens, :]
    
    # Set default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(7)]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(active_attn, cmap='viridis')
    
    # Set labels
    ax.set_title(f'Feature-to-Query Attention (Active Tokens: {active_tokens})')
    ax.set_xlabel('Input Features')
    ax.set_ylabel('Active Query Tokens')
    fig.colorbar(cax, label='Attention Weight')
    
    # Set ticks
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(active_tokens))
    ax.set_xticklabels(feature_names)
    ax.set_yticklabels([f'Token {i}' for i in range(active_tokens)])
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    
    return active_attn, fig

def run_ig_attribution(model, observations, feature_names=None, baseline_type="zero", n_steps=25):
    """
    Compute Integrated Gradients attribution for given observations
    
    Args:
        model: Loaded RL model
        observations: NumPy array of observations (shape: [n_samples, 7])
        feature_names: List of feature names for visualization
        baseline_type: Type of baseline ("zero" or "mean")
        n_steps: Number of steps for IG approximation
    
    Returns:
        attributions: Computed attribution scores (n_samples, n_features)
        fig: Matplotlib figure object
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
    
    # Compute attributions
    attributions = []
    for i, obs in enumerate(obs_tensor):
        obs = obs.unsqueeze(0)  # Add batch dimension
        attr = ig.attribute(obs, 
                           baselines=baseline[i:i+1], 
                           n_steps=n_steps)
        attributions.append(attr.detach().cpu().squeeze(0))
    
    # Convert to numpy
    attr_np = np.array([a.numpy() for a in attributions])
    
    # Average across samples if multiple
    if len(observations) > 1:
        avg_attr = attr_np.mean(axis=0)
    else:
        avg_attr = attr_np[0]
    
    # Set default feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(7)]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['skyblue' if a > 0 else 'salmon' for a in avg_attr]
    ax.bar(range(len(avg_attr)), avg_attr, color=colors)
    ax.set_title("Feature Attribution via Integrated Gradients")
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Attribution Score")
    
    # Set feature names
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    
    return attr_np, fig

# Example usage:
if __name__ == "__main__":
    # 1. Load the model
    model = load_model(
        "pretrained_attention_model",
        policy_type='RecurrentPPO'  # or 'MlpPolicy'
    )
    
    # 2. Create sample observations (7 features)
    observations = np.random.randn(5, 7)  # 5 samples, 7 features each
    
    # 3. Define feature names
    feature_names = [
        "vmAllocatedRatio",
        "avgPUUtilization",
        "avgMemoryUtilization",
        "p90MemoryUtiCPUUtilization",
        "p90Clization",
        "waitingJobsRatioGlobal",
        "waitingJobsRatioRecent"
    ]
    
    # 4. Compute and visualize attention matrix
    attention_matrix, attn_fig = compute_attention_matrix(
        model=model,
        observations=observations,
        feature_names=feature_names
    )
    attn_fig.savefig("attention_matrix.png")
    plt.show(attn_fig)
    
    # 5. Compute and visualize IG attribution
    attributions, ig_fig = compute_ig_attribution(
        model=model,
        observations=observations,
        feature_names=feature_names,
        baseline_type="mean",
        n_steps=50
    )
    ig_fig.savefig("ig_attribution.png")
    plt.show(ig_fig)