import torch
import numpy as np
from typing import Optional
from stable_baselines3 import PPO, DQN

from sb3_contrib import RecurrentPPO
from attribute import *
import plot
from ppo_policy import *
from utils import *
from ppo_policy import PPOPolicy, RecurrentPPOPolicy

class DRLAgentInterpreter:
    def __init__(self,
                 drl_agent_type: str,
                 dnn_model_architecture: str,
                 agent_path: str,
                 observations_path: str,
                 predictions_path: Optional[str] = None,
                 attribution_method: str = 'IG',
                 n_samples: int = 2000,
                 random_samples: bool = True
                 )-> None:
        self.drl_agent_type = drl_agent_type
        self.dnn_model_architecture = dnn_model_architecture

        self.agent = self.__load_agent(agent_path=agent_path)
        self.observations = self.__load_observations(observations_path=observations_path)
        self.predictions = self.__make_predictions(predictions_path=predictions_path)
        self.dnn_model = self.__extract_dnn_model()
        self.feature_attributor = self.__build_feature_attributor(attribution_method=attribution_method,
                                                                  n_samples=n_samples,
                                                                  random_samples=random_samples)

    def __load_agent(self, agent_path: str):
        if self.drl_agent_type == 'DQN':
            return DQN.load(agent_path)
        elif self.drl_agent_type == 'PPO':
            return PPO.load(agent_path)
        elif self.drl_agent_type == 'RecurrentPPO':
            return RecurrentPPO.load(agent_path)
        else:
            raise ValueError

    def __load_observations(self, observations_path: str):
        return torch.Tensor(np.load(observations_path))
    
    def __make_predictions(self, predictions_path: Optional[str]):
        if predictions_path:
            return np.squeeze(np.load(predictions_path), axis=1)
        else:
            predictions = []
            for _ in self.observations:
                predictions.append(self.agent.predict(self.observations[0])[0][0])
            return np.array(predictions)
    
    def __extract_dnn_model(self):
        if self.drl_agent_type == 'DQN':
            return self.agent.q_net
        elif self.drl_agent_type == 'PPO':
            return PPOPolicy(self.agent.observation_space,
                             self.agent.action_space,
                             self.agent.lr_schedule,
                             self.agent)
        elif self.drl_agent_type == 'RecurrentPPO':
            return RecurrentPPOPolicy(self.agent.observation_space,
                                      self.agent.action_space,
                                      self.agent.lr_schedule,
                                      self.agent)
        else:
            raise ValueError
    
    def __build_feature_attributor(self,
                                   attribution_method: str,
                                   n_samples: int,
                                   random_samples: bool):
        if attribution_method == 'IG':
            N = n_samples if self.observations.shape[0] > n_samples else self.observations.shape[0]
            if random_samples:
                idxs = np.random.choice(np.arange(self.observations.shape[0]), size=N)
            else:
                idxs = np.arange(N)
            X_sample = self.observations[idxs]
            predictions_sample = self.predictions[idxs]

            return IGAttributor(net=self.dnn_model,
                                policy=self.dnn_model_architecture.lower(),
                                agent=self.drl_agent_type.lower(),
                                data=X_sample,
                                predictions=predictions_sample)

    def plot_action_histogram(self):
        plot.plot_action_histogram(predictions=self.predictions,
                                   title=f'Action histogram for {self.drl_agent_type}-{self.dnn_model_architecture} agent')
        
    def plot_allocation_vs_queue(self, observations = None):
        if self.dnn_model_architecture == 'CNN':
            X = self.observations
            X = X[:, :, :, :1, :]
            X = np.squeeze(X, axis=1)
            X = np.squeeze(X, axis=1)
            X = np.squeeze(X, axis=1)
        elif self.dnn_model_architecture == 'MLP':
            X = np.squeeze(self.observations, axis=1)
        plot.plot_allocation_vs_queue(observations=X,
                                      title=f'Example run of {self.drl_agent_type}-{self.dnn_model_architecture} agent')

    def plot_mean_attributions(self, absolute_values: bool = False, title_postfix: str = ''):
        mean_attributions = self.feature_attributor.get_mean_attributions()
        plot.plot_mean_attributions(mean_attributions=mean_attributions[0],
                                    policy=self.dnn_model_architecture.lower(),
                                    abs=absolute_values,
                                    title_postfix=title_postfix)
