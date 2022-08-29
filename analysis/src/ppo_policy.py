from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

class PPOPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, model):
        super(PPOPolicy, self).__init__(observation_space, action_space, lr_schedule)
        self.features_extractor = model.policy.features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net
        self.value_net = model.policy.value_net
        
    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        actions = self.action_net(latent_pi)
        return values


from torch import nn
import torch
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy, RecurrentActorCriticCnnPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

class RecurrentPPOPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, model):
        super().__init__(observation_space, action_space, lr_schedule)
        self.features_extractor = model.policy.features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net
        self.value_net = model.policy.value_net
        self.lstm_actor = model.policy.lstm_actor
        self.lstm_critic = model.policy.lstm_critic

        
    def forward(
        self,
        obs,
        lstm_states: RNNStates,
        episode_starts,
        deterministic: bool = False
        ):
        features = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)
        # return torch.Tensor([actions, values, log_prob])
        return values



class RecurrentPPOCnnPolicy(RecurrentActorCriticCnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, model):
        super().__init__(observation_space, action_space, lr_schedule)
        self.features_extractor = model.policy.features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net
        self.value_net = model.policy.value_net
        self.lstm_actor = model.policy.lstm_actor
        self.lstm_critic = model.policy.lstm_critic

    def forward(
        self,
        obs,
        lstm_states: RNNStates,
        episode_starts,
        deterministic: bool = False
        ):
        # features = obs
        features = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # return torch.Tensor([actions, values, log_prob])
        return log_prob
