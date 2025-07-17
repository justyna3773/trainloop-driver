import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from attention_feature_selection import AttentionFeatureExtractor,visualize_selected_features
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from pca_feature_extractor import compute_pca, select_features_by_components, visualize_pca



class FeatureSelector:
    """Unified feature selection interface with multiple methods"""
    def __init__(self, method='attention', env=None, args=None, tensorboard_log=None, total_timesteps=10000, feature_names=None, algo="RecurrentPPO", callback=None):
        """
        Initialize feature selector
        
        :param method: 'attention', 'pca'
        :param model: Trained RL model (required for 'attention')
        :param n_components: Number of components for PCA
        """
        self.method = method
        self.env = env
        self.args = args
        self.tensorboard_log = tensorboard_log
        self.total_timesteps = total_timesteps
        self.model = None  # Placeholder for RL model
        self.feature_names = feature_names if feature_names is not None else []
        self.algo = algo
        self.policy = self.args.policy
        self.callback = callback
        # Initialize sub-selectors
        self.pca = None

        
    def fit(self):
        """Fit feature selectors that require training data (PCA, KBest)"""
        if self.method == 'attention':
            policy_kwargs = {
            "features_extractor_class": AttentionFeatureExtractor,
            "features_extractor_kwargs": {"ne": 32, "input_dim": 15},  # Adjust NE as needed
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])] # Policy/value networks
        }

            # Initialize PPO agent
            if self.algo == "RecurrentPPO":
                print("Using RecurrentPPO with Attention Feature Extractor")
                model = RecurrentPPO(
                            self.policy,
                            self.env,
                            policy_kwargs=policy_kwargs,
                            learning_rate=0.0001, # 0.00003
                            vf_coef=1,
                            clip_range_vf=10.0,
                            max_grad_norm=1,
                            gamma=0.95,
                            ent_coef=0.001,
                            clip_range=0.05,
                            verbose=1,
                            seed=int(self.args.seed),
                            tensorboard_log=self.tensorboard_log)
                self.model = model
            else:
                print("Using PPO with Attention Feature Extractor")
                model = PPO(
                        self.policy,
                        self.env,
                        policy_kwargs=policy_kwargs,
                        learning_rate=0.0001, # 0.00003
                        vf_coef=1,
                        clip_range_vf=10.0,
                        max_grad_norm=1,
                        gamma=0.95,
                        ent_coef=0.001,
                        clip_range=0.05,
                        verbose=1,
                        seed=int(self.args.seed),
                        tensorboard_log=self.tensorboard_log)
                self.model = model
            print(f'Training Attention Feature Extractor for {self.total_timesteps} timesteps...')
            self.model.learn(total_timesteps=self.total_timesteps, callback=self.callback)  # Adjust timesteps as needed
        try:
            visualize_selected_features(model, self.env.observation_space.sample(), feature_names=self.feature_names)
        except:
            pass
        if self.method=='pca':
            # Compute PCA and visualize
            pca_components, pca, observations = compute_pca(self.env, n_samples=10000, visualize=True, n_components=4)
            visualize_pca(pca, observations, self.feature_names )
            # Select features using PCA loadings
            self.feature_mask = select_features_by_components(
                pca.components_,
                threshold=0.5,
                method='max_loading'
            )
            print(f"Selected feature indices: {self.feature_mask}")
            feature_mask = [0 for n in range(0, 7)]
            for n in self.feature_mask:
                feature_mask[n] = 1
            self.feature_mask = feature_mask
            model = None
        self.model = model

    
    def get_feature_mask(self):
        """
        Get feature selection mask or weights
        Returns full feature set for non-attention methods
        
        :return: Attention weights or selection mask
        """
        from run import test_model
        if self.method == 'attention' and self.feature_names is not None:
            new_policy_total_reward, observations, episode_lenghts, rewards, rewards_per_run, observations_evalute_results, actions = test_model(self.model, self.env, n_runs=15)
            try:
                feature_mask = visualize_selected_features(self.model, observations[0], feature_names=self.feature_names)
            except:
                pass
            return feature_mask
        if self.method == 'pca':
            return self.feature_mask
    def get_model(self):
        return self.model

import numpy as np
from gym import ObservationWrapper

# class FeatureMaskWrapper(ObservationWrapper):
#     def __init__(self, env, feature_mask):
#         super().__init__(env)
#         self.feature_mask = np.array(feature_mask, dtype=np.float32)
        
#         # Validate mask dimensions
#         assert len(feature_mask) == env.observation_space.shape[0], \
#             "Mask length must match number of features"
        
#     def observation(self, obs):
#         return obs * self.feature_mask  # Apply element-wise multiplication

from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np

class FeatureMaskWrapper(VecEnvWrapper):
    """Automatically handles action masks for vectorized environments"""
    def __init__(self, venv, feature_mask, policy=None):
        super().__init__(venv)
        if policy=='CnnPolicy':
            self.feature_mask = np.array(feature_mask, dtype=np.float32).reshape(1, 1, 15,7)
        else:
            self.feature_mask = np.array(feature_mask, dtype=np.float32).reshape(1, 7)
        #reshape(1,7)
        # self.mask_buffer = np.ones(
        #     (self.num_envs, self.action_space.n), 
        #     dtype=bool
        # )

    def reset(self):
        obs = self.venv.reset()
        obs = obs*self.feature_mask
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = obs * self.feature_mask
        # # Update masks from infos
        # for i, info in enumerate(infos):
        #     if "action_mask" in info:
        #         self.mask_buffer[i] = info["action_mask"]
                
        return obs, rewards, dones, infos

    def get_action_mask(self):
        """Get current mask for all environments"""
        return self.feature_mask.copy()