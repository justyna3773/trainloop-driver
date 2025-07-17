from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import numpy as np
from stable_baselines3 import PPO
import joblib

from pca_feature_extractor import TruePCAFeatureExtractor, compute_pca, visualize_pca
import autoencoder as ae
feature_names = [ "vmAllocatedRatio",
            "avgPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtiCPUUtilization",
            "p90Clization",
            "waitingJobsRatioGlobal",
            "waitingJobsRatioRecent"] # example
class DimensionReductor(BaseFeaturesExtractor):
    def __init__(self, env, args, reduction_type='attention', reduced_dim=32, 
                hidden_dim=64, tensorboard_log=".", pca_components=None, callback=None, total_timesteps=10_000):
        """
        observation_space: Gym observation space
        reduction_type: 'attention' or 'pca'
        reduced_dim: Output dimension after reduction
        pca_components: Precomputed PCA matrix (reduced_dim x input_dim)
        hidden_dim: Hidden layer size for attention (ignored for PCA)
        """
        input_dim = env.observation_space.shape[0]
        super().__init__(env.observation_space, features_dim=reduced_dim)
        self.reduction_type = reduction_type
        self.reduced_dim = reduced_dim
        self.env = env
        self.model = None
        self.tensorboard_log = tensorboard_log
        self.initial_callback = callback
        self.total_timesteps = total_timesteps
        if reduction_type == 'pca':
            pca_components = compute_pca(env, n_samples=10_000, visualize=False)

                        # Initialize with PCA feature extractor
            model = PPO(
                policy="MlpPolicy",
                env=env,
                policy_kwargs={
                    "features_extractor_class": TruePCAFeatureExtractor,
                    "features_extractor_kwargs": {
                        "pca_components": pca_components  # From step 1
                    }
                },
                verbose=1
            )
            pca_components, pca, observations = compute_pca(env, n_samples=10_000, visualize=True)
            visualize_pca(pca, observations, feature_names=feature_names)
            self.model = model
            
        elif reduction_type == 'attention':
            #TODO this is for temporal_spatial attention
            from attention_rl_automatic_detachable import train_model
            model, _ = train_model(env, policy_type='MlpPolicy', total_timesteps=self.total_timesteps, tensorboard_log=self.tensorboard_log, callback=self.initial_callback)
            #from attention_rl_automatic_detachable import pretrain_feature_extractor
            #model = pretrain_feature_extractor(env)
            self.model = model
            # from set_transformer import train_model
            # print(env.observation_space.shape[0])

            # model = train_model(env)
            # self.model = model
            # from attention_feature_extraction_2 import train_model
            # model = train_model(env)
            # self.model = model
            #best_dim, best_model = find_optimal_dimension_attention(env, args)
            #self.model = best_model
        elif reduction_type == 'recurrentattention':
            from attention_rl_automatic_detachable import train_model
            model, _ = train_model(env, policy_type='RecurrentPPO', total_timesteps=self.total_timesteps, tensorboard_log=self.tensorboard_log, callback=self.initial_callback)
            self.model = model
        elif reduction_type == 'attention_temporal_spatial':
            #from attention_feature_extraction_4 import train_model
            #from attention_rl_automatic_temporal_spatial import train_model
            from attention_rl_temporal_spatial_detachable import train_model
            model = train_model(env, tensorboard_log=self.tensorboard_log, total_timesteps=self.total_timesteps, callback=self.initial_callback)
            #model = pretrain_feature_extractor(env)
            self.model = model
        elif reduction_type=='autoencoder':
            # model = ae.AEPolicy("MlpPolicy",env=env,
            #          learning_rate=0.00003, # 0.00003
            #          vf_coef=1,
            #          clip_range_vf=10.0,
            #          max_grad_norm=1,
            #          gamma=0.95,
            #          ent_coef=0.001,
            #          clip_range=0.05,
            #          verbose=1,
            #          seed=int(args.seed),
            #          tensorboard_log='./autoencoder_log/')
            # Initialize policy
            model = ae.AEPolicy(
                ae.CustomActorCriticPolicy,
                env=env,  # Required parameter
                max_latent_dim=8,
                beta=5.0,
                target_kl=2.5, 
                # Optional parameters:
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64
            )

            ae.pretrain_ae(env, model)
            model.freeze_encoder()  # Critical: Freeze after pretraining
            model.learn(total_timesteps=50_000)
# Continue with RL training
            #model.learn(total_timesteps=10_000)
            self.model = model
        elif reduction_type=='sparse_autoencoder':
            from sparse_autoencoder import train_sparse_ae_model
            self.model = train_sparse_ae_model(env, tensorboard_log=self.tensorboard_log, callback=self.initial_callback)
        elif reduction_type=='sparse_autoencoder_temporal_spatial':
            from sparse_autoencoder_3 import train_autoencoder, continue_training
            self.model = continue_training(self.env, feature_names=feature_names, tensorboard_log=self.tensorboard_log, callback=self.initial_callback)

            

    def get_model(self):
        return self.model
    def get_callback(self):
        return self.initial_callback
    

def find_optimal_dimension_attention(env, args, dims_to_test=[2,4], threshold_difference=0.05):    
    from attention_feature_extraction import AttentionFeatureExtractor, plot_attention_weights, visualize_attention
    best_dim = 3
    best_reward = -float('inf')
    best_model = None
    best_model_found=False
    for dim in dims_to_test:
        print(f"Testing dimension: {dim}")
        
        #model = None
        # model = PPO("MlpPolicy", env, 
        #            policy_kwargs={"features_extractor_class": SmartFeatureExtractor,
        #                           "features_extractor_kwargs": {"max_reduced_dim": dim}})
        policy_kwargs = {
            "features_extractor_class": AttentionFeatureExtractor,
            "features_extractor_kwargs": {
                "features_dim": dim,  # Reduced feature dimension
                "key_dim": 8,
                "value_dim": 8
            },
        }

        model = PPO("MlpPolicy",env=env,
                     learning_rate=0.0001, # 0.00003
                     vf_coef=1,
                     clip_range_vf=10.0,
                     max_grad_norm=1,
                     gamma=0.95,
                     ent_coef=0.001,
                     clip_range=0.05,
                     verbose=1,
                     seed=int(args.seed),
                     policy_kwargs=policy_kwargs)
                     #tensorboard_log=tensorboard_log)
        model.learn(total_timesteps=10000)
        new_policy_total_reward, observations, episode_lenghts, rewards, rewards_per_run, observations_evalute_results, actions = test_model(model, env, n_runs=10)
        mean_reward = rewards_per_run.mean()
        print(f"Testing dimension {dim}: Mean reward: {mean_reward}")
        if (mean_reward - best_reward) <= threshold_difference:
            best_reward = mean_reward
            best_dim = dim
            best_model = model
            best_model_found = True
    if not best_model_found:
        best_model = model
    plot_attention_weights(best_model.policy.features_extractor.reducer.attention_weights, feature_names=feature_names)
    visualize_attention(best_model, observations[0], feature_names=feature_names)
        
    return best_dim, best_model

def test_model(model, env, n_runs=6):
    episode_rewards = []
    episode_rewards_per_run = []
    observations = []
    actions = []
    observations_evalute_results = []
    episode_lenghts = []
    cores_count = []
    for i in range(n_runs):
        episode_len = 0
        obs = env.reset()
        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))
        episode_reward = 0
        rewards_run = []
        observations_run = []
        done = False
        while not done:
            if state is not None:
                action, _, state, _ = model.predict(obs, S=state, M=dones)
            else:
                action, _states = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            reward = rew[0]
            episode_reward += reward
            obs_arr = env.render(mode='array')
            obs_last = [serie[-1] for serie in obs_arr]
            print(f'last obs {obs_last} | reward: {rew} | ep_reward: {episode_reward}')
            print(f'mean obs {np.array(obs).mean(axis=1)}')
            observations_run.append(obs)
            observations.append(obs)
            actions.append(action)
            rewards_run.append(reward)
            episode_len += 1

        if isinstance(episode_reward, list):
            episode_reward = episode_reward.sum()
        episode_rewards_per_run.append(episode_reward)
        episode_lenghts.append(episode_len)
        episode_rewards.append(np.array(rewards_run))
        observations_evalute_results.append(np.array(observations_run))

    # print(episode_rewards)
    mean_episode_reward = np.array(episode_rewards_per_run).mean()
    print(f'Mean reward after {n_runs} runs: {mean_episode_reward}')
    return mean_episode_reward, observations, np.array(episode_lenghts), np.array(episode_rewards), np.array(episode_rewards_per_run), np.array(observations_evalute_results), np.array(actions)

def load_dimension_reductor(env, args, model_path=None, state_dict="pretrained_sparse_ae.pth"):

    if args.algo=='Sparse_autoencoder':
        from sparse_autoencoder import AdaptiveSparseAE_FeatureExtractor, AdaptiveAETrainer, pretrain_ae, train_sparse_ae_model
        fe = AdaptiveSparseAE_FeatureExtractor(env.observation_space)
        ae = fe.autoencoder
        import torch
        ae.load_state_dict(torch.load(state_dict))

        # 3. Prepare policy_kwargs with reconstructed autoencoder
        policy_kwargs = {
            "features_extractor_class": AdaptiveSparseAE_FeatureExtractor,
            "features_extractor_kwargs": {
                "max_latent_dim": 8,
                "hidden_dims": [128, 64],
                "sparsity_coeff": 0.1,
                "threshold": 0.2,
                "pretrained_ae": ae
            },
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
        }

        # 4. Load model with custom objects
        model = PPO.load(
            model_path,
            env=env,
            custom_objects={'policy_kwargs': policy_kwargs}
        )
        return model
    if args.algo=='Sparse_autoencoder_temporal_spatial':
        from sparse_autoencoder_3 import CustomFeaturesExtractor, SparseAutoencoder
        import torch
        # Train autoencoder
        encoder_path = 'autoencoder.pth'
        autoencoder = SparseAutoencoder.load(encoder_path)

        autoencoder.to('cpu')

        # Get feature extractor directly from autoencoder
        pretrained_extractor = autoencoder.get_feature_extractor()
        # Freeze the extractor
        for param in pretrained_extractor.parameters():
            param.requires_grad = False
        pretrained_extractor.eval()

        # Create policy with the pretrained extractor
        policy_kwargs = {
            "features_extractor_class": CustomFeaturesExtractor,
            "features_extractor_kwargs": {
                "encoder": pretrained_extractor,
                "latent_dim": 32
            },
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])]
        }

        model = PPO.load(
            model_path,
            env=env,
            custom_objects={'policy_kwargs': policy_kwargs}
        )
        return model
    if args.algo=='PCA':
        pca_loaded = joblib.load('rl_pca_model.joblib')
        pca_components = pca_loaded.components_
        policy_kwargs={
                "features_extractor_class": TruePCAFeatureExtractor,
                "features_extractor_kwargs": {
                    "pca_components": pca_components  # From step 1
                }
            }
        model = PPO.load(
            model_path,
            env=env,
            custom_objects={'policy_kwargs': policy_kwargs}
        )
        return model
    if args.algo=="Attention":
        from attention_rl_automatic import AdaptiveAttentionFeatureExtractor
        loaded_model = PPO.load(
        model_path,
        # For training: pass environment
        env=env,
        # For inference only: omit env
        custom_objects={
            "features_extractor_class": AdaptiveAttentionFeatureExtractor,
            "features_extractor_kwargs": {
            "d_embed": 64,
            "d_k": 32,
            "min_tokens": 2,
            "max_tokens": 8,
            "temperature": 0.5,
            "reg_strength": 0.1
        },
        }
    )
        return loaded_model
    #     from attention_feature_extraction_2 import AdaptiveAttentionFeatureExtractor
    #     loaded_model = PPO.load(
    #     "models/attention_MlpPolicy_model.zip",
    #     # For training: pass environment
    #     env=env,
    #     # For inference only: omit env
    #     custom_objects={
    #         "features_extractor_class": AdaptiveAttentionFeatureExtractor,
    #         "features_extractor_kwargs": {
    #             "max_reduced_dim": 8,
    #             "min_reduced_dim": 2,
    #             "key_dim": 16,
    #             "value_dim": 16,
    #             "temp_init": 1.0,
    #             "temp_decay": 0.995
    #         }
    #     }
    # )
    #     return loaded_model
        # from attention_feature_extraction import AttentionFeatureExtractor
        # dim = 5
        # policy_kwargs = {
        #     "features_extractor_class": AttentionFeatureExtractor,
        #     "features_extractor_kwargs": {
        #         "features_dim": dim,  # Reduced feature dimension
        #         "key_dim": 8,
        #         "value_dim": 8
        #     },
        # }
        # model = PPO.load(
        #     "models/attention_MlpPolicy_model.zip",
        #     env=env,
        #     custom_objects={'policy_kwargs': policy_kwargs}
        # )
        # return model
        pass


from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# class RewardTrackerCallback(BaseCallback):
#     """
#     Tracks episode rewards across multiple training sessions
#     """
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.episode_rewards = []
#         self.episode_lengths = []
#         self.episode_timesteps = []
#         self.current_rewards = None
#         self.current_lengths = None
        
#     def _init_callback(self) -> None:
#         # Initialize at start of training
#         self.current_rewards = [[] for _ in range(self.training_env.num_envs)]
#         self.current_lengths = [[] for _ in range(self.training_env.num_envs)]
        
#     def _on_step(self) -> bool:
#         # Update reward accumulators
#         for env_idx in range(self.training_env.num_envs):
#             if self.locals['dones'][env_idx]:
#                 # Record completed episode
#                 ep_reward = sum(self.current_rewards[env_idx])
#                 ep_length = len(self.current_rewards[env_idx])
                
#                 self.episode_rewards.append(ep_reward)
#                 self.episode_lengths.append(ep_length)
#                 self.episode_timesteps.append(self.num_timesteps)
                
#                 # Reset accumulator
#                 self.current_rewards[env_idx] = []
#                 self.current_lengths[env_idx] = []
            
#             # Add current reward (even if episode just ended)
#             self.current_rewards[env_idx].append(self.locals['rewards'][env_idx])
#             self.current_lengths[env_idx].append(1)
            
#         return True
    
#     def get_training_history(self):
#         """Returns full training history"""
#         return {
#             'timesteps': np.array(self.episode_timesteps),
#             'rewards': np.array(self.episode_rewards),
#             'lengths': np.array(self.episode_lengths)
#         }
#     def save_to_csv(self, filename="training_rewards.csv"):
#         """
#         Save all collected training data to a CSV file using pandas
#         """
#         # Create DataFrame
#         import pandas as pd
#         df = pd.DataFrame({
#             'timestep': self.episode_timesteps,
#             'reward': self.episode_rewards,
#             'length': self.episode_lengths
#         })
        
#         # Add episode numbers
#         df.insert(0, 'episode', range(1, len(df) + 1))
        
#         # Save to CSV
#         df.to_csv(filename, index=False)
#         return df

from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import os

class RewardTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.timesteps = []
        self.training_data = []
    
    def _on_step(self) -> bool:
        """Required by BaseCallback, but we don't need to do anything here"""
        return True
    
    def on_training_end(self, training_data=None, **kwargs) -> None:
        """Called at the end of training to receive final data"""
        if training_data is not None:
            self.rewards += training_data[0]
            self.timesteps += training_data[1]
        
    
    def save_to_csv(self, filename="training_history.csv"):
        """Save the received training data to CSV"""
        
        print(self.rewards)
        print(self.timesteps)
        print(f'Len rewards: {len(self.rewards)}, len of timesteps: {len(self.timesteps)}')
        if (len(self.timesteps) - len(self.rewards))==1:
            # If there's one extra timestep, remove it
            self.timesteps = self.timesteps[:-1]
        # Create DataFrame
        df = pd.DataFrame({
            'timestep': self.timesteps,
            'reward': self.rewards,
        })
        
        # Save to CSV
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        except:
            pass
        df.to_csv(filename, index=False)
        return df

# from collections import deque
# import numpy as np
# from stable_baselines3.common.callbacks import BaseCallback
# import pandas as pd
# import os

# class RewardTrackerCallback(BaseCallback):
#     """
#     Tracks episode rewards and metrics matching console outputs
#     """
#     def __init__(self, verbose=0, window_size=100):
#         super().__init__(verbose)
#         self.window_size = window_size
        
#         # Store episode-based metrics
#         self.episode_data = {
#             'episode': [],
#             'timestep': [],
#             'reward': [],         # Individual episode reward
#             'length': [],         # Individual episode length
#             'ep_rew_mean': [],    # The exact rolling mean from console
#             'ep_len_mean': []     # The exact rolling mean from console
#         }
        
#         # For accumulating current episode
#         self.current_rewards = None
#         self.current_lengths = None
        
#         # Store step-based metrics
#         self.step_metrics = {
#             'timestep': [],
#             'time/fps': [],
#             'time/iterations': [],
#             'time/time_elapsed': [],
#             'time/total_timesteps': [],
#             'train/approx_kl': [],
#             'train/clip_fraction': [],
#             'train/clip_range': [],
#             'train/clip_range_vf': [],
#             'train/entropy_loss': [],
#             'train/explained_variance': [],
#             'train/learning_rate': [],
#             'train/loss': [],
#             'train/n_updates': [],
#             'train/policy_gradient_loss': [],
#             'train/value_loss': []
#         }
        
#     def _init_callback(self) -> None:
#         self.current_rewards = [[] for _ in range(self.training_env.num_envs)]
#         self.current_lengths = [[] for _ in range(self.training_env.num_envs)]
        
#     def _on_step(self) -> bool:
#         # Update reward accumulators
#         for env_idx in range(self.training_env.num_envs):
#             # Add current reward
#             self.current_rewards[env_idx].append(self.locals['rewards'][env_idx])
#             self.current_lengths[env_idx].append(1)
            
#             if self.locals['dones'][env_idx]:
#                 # Calculate individual episode metrics
#                 ep_reward = np.sum(self.current_rewards[env_idx])
#                 ep_length = len(self.current_rewards[env_idx])
                
#                 # Record individual episode metrics
#                 self.episode_data['episode'].append(len(self.episode_data['episode']) + 1)
#                 self.episode_data['timestep'].append(self.num_timesteps)
#                 self.episode_data['reward'].append(ep_reward)
#                 self.episode_data['length'].append(ep_length)
                
#                 # Reset accumulator
#                 self.current_rewards[env_idx] = []
#                 self.current_lengths[env_idx] = []
        
#         # Capture metrics from infos (if available)
#         if 'infos' in self.locals:
#             for info in self.locals['infos']:
#                 # Skip episode termination info
#                 if 'episode' in info:
#                     continue
                
#                 # Create metric record for this timestep
#                 metric_record = {'timestep': self.num_timesteps}
#                 found_metric = False
                
#                 # Capture rollout metrics (ep_rew_mean and ep_len_mean)
#                 if 'rollout/ep_rew_mean' in info:
#                     self.episode_data['ep_rew_mean'].append(info['rollout/ep_rew_mean'])
#                     found_metric = True
                
#                 if 'rollout/ep_len_mean' in info:
#                     self.episode_data['ep_len_mean'].append(info['rollout/ep_len_mean'])
#                     found_metric = True
                
#                 # Capture other metrics
#                 for metric in self.step_metrics.keys():
#                     if metric == 'timestep':
#                         continue
#                     if metric in info:
#                         metric_record[metric] = info[metric]
#                         found_metric = True
                
#                 # Only add record if we found at least one metric
#                 if found_metric:
#                     for metric, value in metric_record.items():
#                         if metric in self.step_metrics:
#                             self.step_metrics[metric].append(value)
            
#         return True
    
#     def save_to_csv(self, filename="training_history.csv"):
#         """
#         Save all collected training data to CSV files
#         """
#         # Create directory if needed
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
        
#         # Save episode data
#         data_to_save = dict((key,value) for key, value in self.episode_data.items() if key in ['timestep', 'reward', 'length'])
#         df_episode = pd.DataFrame(data_to_save)
#         df_episode.to_csv(filename, index=False)
        
#         # Save step metrics if any were collected
#         # if any(len(v) > 0 for v in self.step_metrics.values()):
#         #     base, ext = os.path.splitext(filename)
#         #     metrics_filename = f"{base}_metrics{ext}"
#         #     df_metrics = pd.DataFrame(self.step_metrics)
#         #     df_metrics.to_csv(metrics_filename, index=False)
        
#         return df_episode