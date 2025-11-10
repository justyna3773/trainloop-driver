from typing import Optional
import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

def _zeros_lstm_states(n_layers: int, batch_size: int, hidden_size: int, device, dtype=th.float32):
    h = th.zeros((n_layers, batch_size, hidden_size), device=device, dtype=dtype)
    c = th.zeros((n_layers, batch_size, hidden_size), device=device, dtype=dtype)
    return (h, c)

class RecurrentPPOPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, model):
        base = model.policy  # grab sizes BEFORE super().__init__()

        shared = getattr(base, "shared_lstm", True)
        hidden = getattr(base, "lstm_hidden_size", base.lstm_actor.hidden_size)
        layers = getattr(base, "n_lstm_layers",  base.lstm_actor.num_layers)

        # Pass correct sizes into the parent so all its internals (including
        # lstm_hidden_state_shape) are created with the desired shapes.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            lstm_hidden_size=hidden,
            n_lstm_layers=layers,
            shared_lstm=shared,
        )

        # Reuse trained submodules
        self.features_extractor = base.features_extractor
        self.mlp_extractor      = base.mlp_extractor
        self.action_net         = base.action_net
        self.value_net          = base.value_net
        self.lstm_actor         = base.lstm_actor
        self.lstm_critic        = base.lstm_critic
        self.action_dist        = base.action_dist
        self.critic             = getattr(base, "critic", None)

        # Mirror meta (not strictly needed now that super got them, but keeps things explicit)
        self.shared_lstm      = shared
        self.lstm_hidden_size = hidden
        self.n_lstm_layers    = layers

        # Ensure the public shape attribute reflects the copied sizes
        # (SB3 sets it in super().__init__, but set again to be explicit)
        self.lstm_hidden_state_shape = (self.n_lstm_layers, 1, self.lstm_hidden_size)

    def get_initial_lstm_states(self, batch_size: int, device: Optional[th.device] = None) -> RNNStates:
        if device is None:
            device = next(self.parameters()).device
        H, L = self.lstm_hidden_size, self.n_lstm_layers
        pi = _zeros_lstm_states(L, batch_size, H, device)
        if self.lstm_critic is not None and not self.shared_lstm:
            vf = _zeros_lstm_states(L, batch_size, H, device)
        else:
            vf = pi
        return RNNStates(pi=pi, vf=vf)

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ):
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
            latent_vf = self.critic(features) if self.critic is not None else features
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self.value_net(latent_vf)
        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)

        return actions, values, log_prob, RNNStates(pi=lstm_states_pi, vf=lstm_states_vf)
