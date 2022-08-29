from email.mime import base
from captum.attr import IntegratedGradients
from utils import Utils
from plot import *
import numpy as np
import torch

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

    def explain_example(self, idx, plot_env=True, all_actions=False, print_q_values=True):
        if self.agent == 'dqn':
            if self.policy == 'mlp':
                self._explain_example_mlp(idx, plot_env, all_actions, print_q_values)
            elif self.policy == 'cnn':
                return self._explain_example_cnn(idx, all_actions, print_q_values)
        elif self.agent == 'ppo':
            if self.policy == 'mlp':
                self._explain_example_mlp_ppo(idx, plot_env)
            elif self.policy == 'cnn':
                return self._explain_example_cnn_ppo(idx)


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


    def _explain_example_mlp_ppo(self, idx, plot_env=True):
        if idx > self.data.shape[0]:
            raise ValueError('Idx exceeds data size')
        x = self.data[idx][0].numpy()

        action_made = self.predictions[idx]
        # print(f'Action made: {self.action_names[action_made]}')

        print(len(self.attributions))

        plot_attribution(idx=idx, action=action_made, attributions=self.attributions[0], img=x)

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
        if self.policy == 'cnn':
            self.attributions_normalized = [action_att.mean(0) for action_att in self.attributions]
        else:
            self.attributions_normalized = [action_att.sum(0) / np.linalg.norm(action_att.sum(0), ord=1) for action_att in self.attributions]
