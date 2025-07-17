from email.mime import base
from captum.attr import IntegratedGradients
from utils import Utils
from plot import *
import numpy as np
import torch
from sparse_autoencoder import compute_feature_importance

class IGAttributor:
    def __init__(
        self,
        net,
        policy,
        agent,
        data,
        predictions,
        method=IntegratedGradients,
        baselines=None,
        target=None,
        # multiply_by_inputs=True,
        forward_args=None,
        n_steps=50,
        attributions_len=1000,
        action_names=Utils.ACTION_NAMES,
        feature_names=Utils.FEATURE_NAMES
    ) -> None:
        self.net = net
        self.policy = policy
        self.agent = agent
        self.target = target
        self.baselines = baselines
        self.forward_args=forward_args
        self.ig = method(net)
        self.n_steps = n_steps
        self.action_names = action_names
        self.attributions = []
        self.delta = []
        self.data = data
        self.predictions = predictions
        self._calculate_attributions_per_action()

    def get_mean_attributions(self) -> list:
        # return self.attributions_normalized
        return [action_att.mean(0) for action_att in self.attributions]

    def explain_example(self, idx, plot_env=True, all_actions=False, print_q_values=True, title_postfix=''):
        if self.agent == 'dqn':
            if self.policy == 'mlp':
                self._explain_example_mlp(idx, plot_env, all_actions, print_q_values)
            elif self.policy == 'cnn':
                return self._explain_example_cnn(idx, all_actions, print_q_values)
        elif self.agent == 'ppo':
            if self.policy == 'mlp':
                self._explain_example_mlp_ppo(idx, plot_env, title_postfix=title_postfix)
            elif self.policy == 'cnn':
                return self._explain_example_cnn_ppo(idx)
        elif self.agent == 'attention':
            self._explain_example_mlp_ppo(idx, plot_env, title_postfix=title_postfix)


    def _explain_example_cnn(self, idx, all_actions, print_q_values):
        img = self.data[idx]

        q_values = self.net.forward(img).detach().numpy()
        action_made = np.argmax(q_values)
        if print_q_values:
            print(f'Action made: {self.action_names[action_made]}')
            print(f'Q-values:')
            for action in range(len(self.action_names)):
                print(f'{self.action_names[action]}: {q_values[0][action]}')
            print()

        plot_attribution_cnn(idx=idx, attributions=self.attributions[action_made], action=action_made, data=self.data)
        if all_actions: 
            plot_attributions_cnn(idx=idx, attributions=self.attributions, action=action, data=self.data)

    def _explain_example_cnn_ppo(self, idx):
        img = self.data[idx]

        action = self.predictions[idx]
        # print(f'Action made: {self.action_names[action]}')

        return plot_attribution_cnn(idx=idx, attributions=self.attributions[0], action=action, data=self.data)

    def _explain_example_mlp(self, idx, plot_env, all_actions, print_q_values):
        if idx > self.data.shape[0]:
            raise ValueError('Idx exceeds data size')
        x = self.data[idx][0].numpy()
        
        q_values = self.net.forward(self.data[idx]).detach().numpy()
        action_made = np.argmax(q_values)

        # print(f'Action made: {self.action_names[action_made]}')
        if print_q_values:
            print(f'Q-values:')
            for action in range(len(self.action_names)):
                print(f'{self.action_names[action]}: {q_values[0][action]}')

        plot_attribution(idx=idx, action=action_made, attributions=self.attributions[action_made], img=x)

        if all_actions:
            plot_attributions(idx=idx, attributions=self.attributions, img=x)


    def _explain_example_mlp_ppo(self, idx, plot_env=True, title_postfix=''):
        if idx > self.data.shape[0]:
            raise ValueError('Idx exceeds data size')
        x = self.data[idx][0].numpy()

        action_made = self.predictions[idx]
        # print(f'Action made: {self.action_names[action_made]}')

        plot_attribution(idx=idx, action=action_made, attributions=self.attributions[0], img=x, title_postfix=title_postfix)

    def _calculate_attributions_per_action(self):
        if len(self.data.shape) == 5:
            data = torch.squeeze(self.data, 1)
        else:
            data = self.data
        if self.agent == 'dqn':
            print(data.shape)
            for target in range(len(self.action_names)):
                attributions, delta = self.ig.attribute(data,
                                                        # n_steps=self.n_steps,
                                                        baselines=self.baselines,
                                                        target=target,
                                                        return_convergence_delta=True)
                # if self.policy == 'cnn':
                #     self.attributions.append(attributions.detach().squeeze(1).numpy())
                # else:
                self.attributions.append(attributions.detach().numpy())
                self.delta.append(delta.detach().numpy())
        if self.agent == 'ppo':
            if self.target is None:
                attributions, delta = self.ig.attribute(data,
                                                        # n_steps=self.n_steps,
                                                        baselines=self.baselines,
                                                        return_convergence_delta=True,
                                                        additional_forward_args=self.forward_args)
            else:
                attributions, delta = self.ig.attribute(data,
                                                        # n_steps=self.n_steps,
                                                        baselines=self.baselines,
                                                        target=self.target,
                                                        return_convergence_delta=True,
                                                        additional_forward_args=self.forward_args)
            self.attributions.append(attributions.detach().numpy())
        if self.agent == 'attention':
            # 1. Freeze adaptive components
            self.net.policy.features_extractor.freeze_adaptation()

            # 2. Wrap model for Captum compatibility
            def forward_wrapper(obs):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    return self.net.policy(obs_tensor)[0]  # Use action logits

            # 3. Compute attributions
            ig = IntegratedGradients(forward_wrapper)
            self.ig = ig
            if self.target is None:
                attributions, delta = self.ig.attribute(data,
                                                        # n_steps=self.n_steps,
                                                        baselines=self.baselines,
                                                        return_convergence_delta=True,
                                                        additional_forward_args=self.forward_args)
            else:
                attributions, delta = self.ig.attribute(data,
                                                        # n_steps=self.n_steps,
                                                        baselines=self.baselines,
                                                        target=self.target,
                                                        return_convergence_delta=True,
                                                        additional_forward_args=self.forward_args)
            self.attributions.append(attributions.detach().numpy())
       
        if self.agent == 'sparse_autoencoder':
            autoencoder = self.net.features_extractor.autoencoder
    
            # Compute feature importance using autoencoder only
            importance_matrix, _ = compute_feature_importance(
                autoencoder=autoencoder,
                observations=self.data,
                n_steps=self.n_steps
            )
            
            # Store results (feature importance matrix)
             # Convert to tensor with shape (n_samples, 1, 7)
            n_samples = importance_matrix.shape[0]
            attributions_tensor = importance_matrix.reshape(n_samples, 1, -1)
            # compute_combined_ig(autoencoder, self.net, self.data, self.predictions)

            self.attributions.append(importance_matrix)

        if self.policy == 'cnn':
            self.attributions_normalized = [action_att.mean(0) for action_att in self.attributions]
        else:
            self.attributions_normalized = [action_att.sum(0) / np.linalg.norm(action_att.sum(0), ord=1) for action_att in self.attributions]


def compute_autoencoder_ig_per_feature(autoencoder, observations, n_steps=50):
    autoencoder.eval()
    device = next(autoencoder.parameters()).device
    
    # Prepare data
    obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
    n_features = obs_tensor.shape[1]
    
    # Correct baseline
    feature_quantiles = torch.quantile(obs_tensor, 0.1, dim=0)
    baseline = feature_quantiles.unsqueeze(0).expand_as(obs_tensor)
    
    # Initialize importance matrix
    importance_matrix = np.zeros((n_features, n_features))
    
    # Compute IG for each output feature separately
    for out_dim in range(n_features):
        def forward_func(inputs):
            reconstructions = autoencoder(inputs)
            return reconstructions[:, out_dim]  # Specific output feature
        
        ig = IntegratedGradients(forward_func)
        attr = ig.attribute(
            inputs=obs_tensor,
            baselines=baseline,
            target=None,  # Scalar output
            n_steps=n_steps
        )
        # Store average importance for this output feature
        importance_matrix[out_dim] = torch.mean(torch.abs(attr), dim=0).cpu().numpy()
    
    return importance_matrix
def compute_combined_ig(autoencoder, policy, observations, actions, n_steps=50):
    """
    Compute IG for autoencoder + policy architecture
    :param autoencoder: Trained autoencoder model
    :param policy: MLP policy network (e.g., MlpPolicy)
    :param observations: Input data (tensor)
    :param actions: Actions taken by policy (targets)
    :param n_steps: IG computation steps
    :return: Feature importance matrix
    """
    # Set models to evaluation mode
    autoencoder.eval()
    policy.eval()
    
    device = next(policy.parameters()).device
    
    # Convert observations to tensor if needed
    if not isinstance(observations, torch.Tensor):
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
    else:
        obs_tensor = observations.to(device)
    
    # Convert actions to tensor and move to device
    if not isinstance(actions, torch.Tensor):
        action_targets = torch.tensor(actions, dtype=torch.long).to(device)
    else:
        action_targets = actions.to(device)
    
    # Create baseline (feature-wise 10th percentile)
    feature_quantiles = torch.quantile(obs_tensor, 0.1, dim=0)
    baseline = feature_quantiles.unsqueeze(0).expand_as(obs_tensor)
    
    # Define forward pass through full system
    def full_forward(inputs):
        # Autoencoder pass
        with torch.no_grad():  # No grad for autoencoder if frozen
            features = autoencoder.encoder(inputs)
            latent = autoencoder.projection(features)
        
        # Policy pass - get distribution and extract logits
        # Handle different latent dimensions
        if latent.dim() == 3:
            latent = latent.squeeze(1)
        dist = policy.get_distribution(latent)
        return dist.distribution.logits
    
    # Compute IG
    ig = IntegratedGradients(full_forward)
    attributions = ig.attribute(
        inputs=obs_tensor,
        baselines=baseline,
        target=action_targets,  # Explain specific actions taken
        n_steps=n_steps
    )
    
    return attributions.detach().cpu().numpy()