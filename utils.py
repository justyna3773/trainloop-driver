# FEATURE_NAMES = [ "vmAllocatedRatio",
#             "avgCPUUtilization",
#             "avgMemoryUtilization",
#             "p90MemoryUtilization",
#             "p90CPUUtilization",
#             "waitingJobsRatioGlobal",
#             "waitingJobsRatioRecent"]

FEATURE_NAMES = ["vmAllocatedRatio",
        "avgCPUUtilization",
        "p90CPUUtilization",
        "avgMemoryUtilization",
        "p90MemoryUtilization",
        "waitingJobsRatioGlobal",
        "waitingJobsRatioRecent"]
PCA_OFFLINE=False
PCA=False


def warmup_linear_schedule(lr0, warmup_frac=0.05):
    # progress goes 1 -> 0 over training in SB3
    def f(progress):
        # during warmup (first warmup_frac), ramp from 0 -> lr0
        warm = (1.0 - progress) < warmup_frac
        if warm:
            w = (1.0 - progress) / max(1e-8, warmup_frac)
            return lr0 * w
        # then linear decay to 0
        return lr0 * progress
    return f


import numpy as np

def _is_vec_env(env) -> bool:
    return hasattr(env, "num_envs") and hasattr(env, "step_async")



def _scalar(x):
    # Convert reward to a Python float
    x = np.asarray(x)
    return float(np.squeeze(x))

def _done_scalar(x):
    # Convert done to a Python bool
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.shape == ():
        return bool(x)
    # If still not scalar, take first element after flatten
    return bool(x.reshape(-1)[0])

def _to_1d_bool(x):
    # Convert episode_start/done array to shape (num_envs,)
    x = np.asarray(x).astype(bool)
    x = np.squeeze(x)
    if x.ndim == 0:
        # single env
        return np.array([bool(x)], dtype=bool)
    if x.ndim > 1:
        x = x.reshape(-1)
    return x

def evaluate(model, env, n_episodes=15, deterministic=False,
             freeze_vecnorm=True, collect_info=False):
    # keep VecNormalize reward semantics = true env reward
    if hasattr(env, "norm_reward"): 
        env.norm_reward = False
    if freeze_vecnorm and hasattr(env, "training"): 
        env.training = False

    is_vec = hasattr(env, "num_envs")
    if is_vec and env.num_envs != 1:
        raise ValueError("Use a single-env VecEnv for eval.")

    state = None
    ep_returns, observations, actions = [], [], []
    raw_obs_trace, raw_metrics_trace = [], []

    for _ in range(n_episodes):
        obs = env.reset()
        episode_start = np.array([True], dtype=bool) if is_vec else None
        done = False
        ret = 0.0

        while not done:
            # SB3's predict API (handle older versions without episode_start)
            try:
                action, state = model.predict(
                    obs, state=state, episode_start=episode_start,
                    deterministic=deterministic
                )
            except TypeError:
                action, state = model.predict(
                    obs, state=state, deterministic=deterministic
                )

            if is_vec:
                obs, r, d, info = env.step(action)
                observations.append(obs)
                actions.append(action)

                # Normalize shapes
                r_scalar = _scalar(r)
                done_scalar = _done_scalar(d)
                episode_start = _to_1d_bool(d)

                if collect_info:
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    raw_obs_trace.append(i0.get("_raw_obs"))
                    raw_metrics_trace.append(i0.get("_raw_metrics"))
            else:
                obs, r, d, info = env.step(action)
                observations.append(obs)
                actions.append(action)
                r_scalar = float(r)
                done_scalar = bool(d)
                if collect_info:
                    raw_obs_trace.append(info.get("_raw_obs"))
                    raw_metrics_trace.append(info.get("_raw_metrics"))

            ret += r_scalar
            done = done_scalar

        ep_returns.append(ret)
        state = None  # reset hidden state between episodes

    mean_return = float(np.mean(ep_returns)) if ep_returns else 0.0
    if collect_info:
        return mean_return, raw_obs_trace, ep_returns, actions
    else:
        return mean_return, observations, ep_returns, actions


#TODO this is basic evaluate function
def evaluate_sample(args, extra_args):
    # build env
    from run import build_env
    import pandas as pd
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    if 'Cnn' in args.policy:
        args.observation_history_length = 15
    env = build_env(args, extra_args)
    print(env.observation_space)
    
    algo = args.algo
    policy = args.policy
    model_name = f'{algo.lower()}_{policy}_{args.model_name}' if args.model_name else f'{algo.lower()}_{policy}'
    if PCA_OFFLINE:
        results = dict(np.load('pca_results.npz'))
        extra_args['results'] = results
        env = build_env(args, extra_args)
    if PCA:
        from pca import load_model
        model, env = load_model(args, extra_args)

    if env:
        
        evaluation_results = pd.DataFrame([])
        if env is not None:
                        
            if policy=='MlpLstmPolicy':
                #model = RecurrentPPO.load(rf'C:\initial_model\historic\recurrentppo\MlpLstmPolicy\recurrentppo_MlplstmPolicy')
                #SPCA
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\SPCA\pca_MlpLstmPolicy_spca_recurrent', env=env)
                #model = RecurrentPPO.load(r'C:\initial_model\historic_synthetic_dnnevo\RecurrentPPO_14\recurrentppo_MlpLstmPolicy_mlplstm_policy_reduced', env=env)
                #model = RecurrentPPO.load(r'C:\initial_model\historic_synthetic_dnnevo\RecurrentPPO_19_worst\recurrentppo_MlpLstmPolicy_mlplstm_policy_worst', env=env)
                #model = RecurrentPPO.load(r'C:\initial_model\historic_synthetic_dnnevo\RANDOM\BASELINE\recurrentppo_MlpLstmPolicy_mlplstm_all_metrics')
                #model = RecurrentPPO.load(r'C:\initial_model\historic_synthetic_dnnevo\RANDOM\random_1_16_RecurrentPPO_62\recurrentppo_MlpLstmPolicy_mlplstm_zbior1', env=env)
                #model = RecurrentPPO.load(r'C:\initial_model\historic_synthetic_dnnevo\RANDOM\random_2_16_RecurrentPPO_48\recurrentppo_MlpLstmPolicy_mlplstm_random_2_16', env=env)
                #model = RecurrentPPO.load(r'C:\initial_model\historic_synthetic_dnnevo\RANDOM\random_3_16_RecurrentPPO_50\recurrentppo_MlpLstmPolicy_mlplstm_random_3_16', env=env)
                #model = RecurrentPPO.load(r'C:\initial_model\historic_synthetic_dnnevo\RANDOM\random_4_16_RecurrentPPO_58\recurrentppo_MlpLstmPolicy_mlplstm_random_4', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\random_5_16_RecurrentPPO_64\recurrentppo_MlpLstmPolicy_mlplstm_zbior4_powtorka.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MANUAL\vmallocated_avgcpu_recurrentppo_82\recurrentppo_MlpLstmPolicy_mlplstm_vm_avgcpu.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\ATTENTTION\vm_global_avgcpu_recurrentppo_96\recurrentppo_MlpLstmPolicy_mlplstm_attn_top_3.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\ATTENTTION\vm_recent_avgcpu_recurrentppo_97\recurrentppo_MlpLstmPolicy_mlplstm_attn_top_3.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MANUAL\vmallocated_p90mem_recurrentppo_83\recurrentppo_MlpLstmPolicy_mlplstm_vm_p90mem', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MANUAL\p90mem_p90cpu_recurrentppo_87\recurrentppo_MlpLstmPolicy_mlplstm_p90cpu_p90mem.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MANUAL\recent_avgcpu_recurrentppo_89\recurrentppo_MlpLstmPolicy_mlplstm_recent_avgcpu.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MANUAL\SET2_recurrentppo_91\recurrentppo_MlpLstmPolicy_mlplstm_manual_2.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\RETRAINING\recurrentppo_108_0_1_5_2\recurrentppo_MlpLstmPolicy_mlplstm_4_top_ig.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\RETRAINING\recurrentppo_107_0_1_5_6\recurrentppo_MlpLstmPolicy_mlplstm_4_top_spca_attention.zip', env=env)
                #model = RecurrentPPO.load(path=rf'C:\initial_model\{algo.lower()}\{policy}\{algo.lower()}_{policy}')
                #                                     env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\BASELINE\arch_64_64\recurrentppo_MlpLstmPolicy_mlplstm_full_model.zip', env=env)

                #TODO manual choice
                #worst
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MANUAL\recurrentppo_119_2_3_4_worst\recurrentppo_MlpLstmPolicy_mlplstm_bottom_3.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MANUAL\SET2_recurrentppo_91\recurrentppo_MlpLstmPolicy_mlplstm_manual_2.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\SYNTHETIC\2_3_4_700_tys\recurrentppo_MlpLstmPolicy_mlplstm_bottom_3_synthetic.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MANUAL\recurrentppo_119_2_3_4_worst\recurrentppo_MlpLstmPolicy_mlplstm_bottom_3.zip', env=env) 
                #set 2
                #model = RecurrentPPO.load(r'c:\initial_model\recurrentppo\MlpLstmPolicy\recurrentppo_MlpLstmPolicy_mlplstm_manual_2.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\PPO_32_avgcpu_waitingglobal_vm\ppo_MlpPolicy_mlp_avgcpu.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\ATTENTTION\vm_recent_avgcpu\recurrentppo_MlpLstmPolicy_mlplstm_attn_top_3.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\BASELINE\arch_256_100_000\ppo_mlp_intermediate_00_100000.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\RETRAINING\recurrentppo_109_110_0_5\recurrentppo_MlpLstmPolicy_mlplstm_4_top_2_attn.zip', env=env)
                #TODO synthetic
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\SYNTHETIC\0_1_5_6_200_tys\smaller_arch_recurrentppo_106\recurrentppo_MlpLstmPolicy_mlplstm_baseline_synthetic.zip', env=env)

                #TODO automatic choice final experiments
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\PICKED_METRICS\0_1_2_3_5_recurrentppo_157\recurrentppo_MlpLstmPolicy_mlplstm_0_1_2_3_5.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\PICKED_METRICS\0_1_2_recurrentppo_155\recurrentppo_MlpLstmPolicy_mlplstm_vm_avgcpu_p90cpu.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\PICKED_METRICS\3_4_6_recurrentppp_156\recurrentppo_MlpLstmPolicy_mlplstm_avgmem_p90mem_waitingrecent.zip', env=env)

                #TODO attention
                model=RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\FINAL_MODELS\RETRAINING\0_1_2_5_recurrentppo_196\recurrentppo_MlpLstmPolicy_mlplstm_0_1_2_5.zip', env=env)
                #model = RecurrentPPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\ATTENTION\recurrentppo_171_0_00007\attention_MlpLstmPolicy_mlplstm_att_tunedtolstm.zip', env=env)
                pass
            elif PCA:
                pass   
            else:
                pass
                #pass
                #model = PPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\RETRAINING\ppo_153_0_1_2\ppo_MlpPolicy_mlp_0_1_2.zip', env=env)
                #model = PPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\RETRAINING\ppo_152_0_1_5\ppo_MlpPolicy_mlp_0_1_5.zip', env=env)
                #model = PPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\RETRAINING\ppo_151_0_1\ppo_MlpPolicy_mlp_0_1.zip', env=env)
                #model = PPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\RETRAINING\ppo_154_2_3_4\ppo_MlpPolicy_mlp_2_3_4.zip', env=env)


                #model = PPO.load(path=rf'C:\initial_model\{algo.lower()}\{policy}\{model_name}',env=env)
                #model = PPO.load(r'c:\initial_model\historic_synthetic_dnnevo\RANDOM\MLP\0_1_2_ppo_145\ppo_MlpPolicy_mlp_0_1_2.zip', env=env)
                #model = PPO.load(path=rf'C:\Users\ultramarine\Desktop\ppo_magisterka\trainloop_driver_official\trainloop_driver_final\trainloop-driver\output_malota\ppo_mlp_intermediate_02_300000',env=env)
                #model = PPO.load(path=r'C:\Users\ultramarine\Desktop\ppo_magisterka\trainloop_driver_official\trainloop_driver_final\trainloop-driver\output_malota\ppo_gawrl_intermediate_02_300000', env=env)
                #model = PPO.load(r'C:\initial_model\historic_synthetic_dnnevo\PPO_43_baseline\ppo_MlpPolicy_mlp_vm_waiting', env=env)
                
            env.reset()
            #mean_reward, observations, episode_lenghts, rewards, rewards_per_run, observations_evalute_results, actions  = test_model(model, env, n_runs=10)
            
            mean_reward, observations, rewards, actions = evaluate(model, env, n_episodes=10)
            print(f'Mean episode reward from correct eval: {mean_reward}')

            with open(rf'C:\initial_model\ppo\MlpPolicy\observations_{algo.lower()}_{policy}_{args.observation_history_length}.npy', 'wb') as f:
                np.save(f, np.array(observations))

            with open(f'/initial_model/rewards/rewards_{algo.lower()}_{policy}.npy', 'wb') as f:
                np.save(f, np.array(rewards))

            with open(f'/initial_model/actions/actions_{algo.lower()}_{policy}.npy', 'wb') as f:
                np.save(f, np.array(actions))
            evaluation_results = pd.DataFrame(list(zip(observations, rewards, actions)), columns = ['obs', 'rew', 'actions'])

            evaluation_results.to_csv(f'/initial_model/eval_results/{model_name}.csv')

    env.close()
    print('Evaluation ended')




def run_episode(env, policy_fn):
    obs = env.reset()
    ep_ret = 0.0
    while True:
        a = policy_fn(obs)  # e.g., lambda o: ACTION_NOTHING
        obs, r, done, info = env.step([a])
        ep_ret += float(r)
        if done:
            return ep_ret



from typing import Optional, Callable
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False


class FeatureMaskWrapper(gym.Wrapper):
    """
    Masks observations for the agent (zeros out selected metrics along the LAST axis),
    but can expose original values via info["_raw_obs"] (and optional info["_raw_metrics"]).

    Args:
        env: base env with Box observation space
        mask: 1D sequence of {0,1}/bool with length == obs.shape[-1]
        expose_raw: if True, put raw obs into info["_raw_obs"] on step()
        raw_metrics_fn: optional callable(raw_obs: np.ndarray) -> dict to add info["_raw_metrics"]
    """
    def __init__(
        self,
        env,
        mask,
        expose_raw: bool = False,
        raw_metrics_fn: Optional[Callable[[np.ndarray], dict]] = None,
    ):
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("FeatureMaskWrapper requires a Box observation space.")
        if env.observation_space.shape is None or len(env.observation_space.shape) < 1:
            raise ValueError("Observation space must be at least 1-D.")

        self._pre_shape = tuple(env.observation_space.shape)
        feat_dim = int(self._pre_shape[-1])

        # Validate & store mask (1D over the last axis)
        m = np.asarray(mask).astype(bool).ravel()
        if m.size != feat_dim:
            raise ValueError(f"Mask length {m.size} != feature dim {feat_dim}.")
        self.mask_last = m  # shape (feat_dim,)
        self.expose_raw = expose_raw
        self.raw_metrics_fn = raw_metrics_fn

        # Build a broadcast mask shaped like the observation for fast elementwise multiply
        # e.g., pre_shape=(1,15,7) -> broadcast mask shape=(1,1,7)
        self._broadcast_shape = (1,) * (len(self._pre_shape) - 1) + (feat_dim,)
        self._mask_broadcast = self.mask_last.reshape(self._broadcast_shape)

        # Update observation_space: same shape/dtype; masked entries are always zero
        low, high = env.observation_space.low, env.observation_space.high
        if np.isscalar(low) and np.isscalar(high):
            low_arr = np.full(self._pre_shape, low, dtype=np.float32)
            high_arr = np.full(self._pre_shape, high, dtype=np.float32)
        else:
            low_arr = np.asarray(low, dtype=np.float32).copy()
            high_arr = np.asarray(high, dtype=np.float32).copy()

        mask_full = np.broadcast_to(self._mask_broadcast, self._pre_shape)
        # where mask==False -> both low and high become 0
        low_arr[~mask_full] = 0.0
        high_arr[~mask_full] = 0.0

        self.observation_space = spaces.Box(
            low=low_arr, high=high_arr, dtype=env.observation_space.dtype
        )

    def _apply_mask(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        # Ensure broadcast works even if obs has same rank as space
        masked = obs * np.broadcast_to(self._mask_broadcast, obs.shape).astype(obs.dtype)
        return masked

    # Reset: support Gymnasium (obs, info) and Gym (obs)
    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        if _GYMNASIUM:
            obs, info = res
            masked = self._apply_mask(obs)
            return masked, info
        else:
            obs = res
            masked = self._apply_mask(obs)
            return masked

    # Step: support Gymnasium 5-tuple and Gym 4-tuple
    def step(self, action):
        res = self.env.step(action)
        if _GYMNASIUM and isinstance(res, tuple) and len(res) == 5:
            obs, reward, terminated, truncated, info = res
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = res

        raw_obs = np.asarray(obs, dtype=np.float32)
        masked_obs = self._apply_mask(raw_obs)

        if self.expose_raw:
            info = dict(info) if info is not None else {}
            info["_raw_obs"] = raw_obs
            if self.raw_metrics_fn is not None:
                info["_raw_metrics"] = self.raw_metrics_fn(raw_obs)

        if _GYMNASIUM and isinstance(res, tuple) and len(res) == 5:
            return masked_obs, reward, terminated, truncated, info
        else:
            return masked_obs, reward, done, info


class FeatureMaskWrapper_backup(gym.ObservationWrapper):
    """
    Masks features by elementwise-multiplying observations with a 0/1 vector.
    Example mask: [1, 1, 0, 0, 1, 1, 1]  -> features with 0 become 0.
    Supports Box spaces and Dict spaces containing Box subspaces.
    """
    def __init__(self, env: gym.Env, feature_mask):
        super().__init__(env)
        self.mask = np.asarray(feature_mask, dtype=np.float32)
        self.observation_space = self._mask_space(self.env.observation_space)

    # ---- space helpers ----
    def _mask_space(self, space):
        if isinstance(space, spaces.Box):
            return self._mask_box_space(space)
        if isinstance(space, spaces.Dict):
            new_spaces = {}
            for k, sub in space.spaces.items():
                if isinstance(sub, spaces.Box) and sub.shape and sub.shape[-1] == self.mask.shape[-1]:
                    new_spaces[k] = self._mask_box_space(sub)
                else:
                    new_spaces[k] = sub
            return spaces.Dict(new_spaces)
        return space  # leave others unchanged

    def _mask_box_space(self, space: spaces.Box):
        # Make masked dims have [0, 0] bounds to avoid inf*0 issues
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)

        if low.shape and low.shape[-1] == self.mask.shape[-1]:
            # reshape mask to broadcast on last dimension
            shape = (1,) * (low.ndim - 1) + (self.mask.shape[-1],)
            m = self.mask.reshape(shape)
            low = np.where(m == 0.0, 0.0, low)
            high = np.where(m == 0.0, 0.0, high)
            return spaces.Box(low=low, high=high, dtype=space.dtype)
        else:
            return space

    # ---- observation hook ----
    def observation(self, obs):
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                v_arr = np.asarray(v)
                if v_arr.shape and v_arr.shape[-1] == self.mask.shape[-1]:
                    shape = (1,) * (v_arr.ndim - 1) + (self.mask.shape[-1],)
                    out[k] = v_arr * self.mask.reshape(shape)
                else:
                    out[k] = v
            return out
        # plain vector/array
        obs = np.asarray(obs)
        shape = (1,) * (obs.ndim - 1) + (self.mask.shape[-1],)
        return obs * self.mask.reshape(shape)


import gym
import numpy as np
from gym import spaces

class SelectMetricsWrapper(gym.ObservationWrapper):
    """
    Keep only metrics at indices `metric_idxs` along the last axis.
    Example: obs shape (1, 8, 20) and metric_idxs=[0,3,7] -> output shape (1, 8, 3).
    """
    def __init__(self, env: gym.Env, metric_idxs):
        super().__init__(env)
        self.metric_idxs = np.array(metric_idxs, dtype=np.int64)
        assert isinstance(env.observation_space, spaces.Box), "Only Box obs supported"

        # Validate indices
        N = env.observation_space.shape[-1]
        if np.any(self.metric_idxs < 0) or np.any(self.metric_idxs >= N):
            raise IndexError(f"indices must be in [0,{N-1}]")

        # Slice low/high on the last axis to build the new Box space
        low  = np.take(env.observation_space.low,  self.metric_idxs, axis=-1)
        high = np.take(env.observation_space.high, self.metric_idxs, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        # Slice the runtime observation along the last axis
        return np.take(obs, self.metric_idxs, axis=-1)

# SelectMetricsWrapper: keep only chosen metrics along the LAST dimension.
# Works with shapes like (7,), (1, 15, 7), (8, 7), etc.


from typing import Iterable, List, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from gym.spaces import Box


class NoisyMetricAugmentWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that augments a 1-D Box observation with per-step noisy metrics
    placed at user-specified indices in the FINAL augmented vector.

    Example:
        Base obs has 7 features. You want to add 8 noisy features and have them at:
        [0, 1, 9, 10, 11, 12, 13, 14]  -> final length = 7 + 8 = 15
        The 7 base features will be interleaved at all indices NOT in noisy_indices,
        in their original order.

    Invariants:
        - Only supports 1-D Box observations.
        - total_len = base_dim + len(noisy_indices)
        - 0 <= each noisy index < total_len and indices must be unique

    Parameters
    ----------
    env : gym.Env
        Base environment with 1-D Box observation space.
    noisy_indices : Iterable[int]
        Indices (in the final augmented observation) where noise in [0,1] will be placed.
    seed : Optional[int]
        RNG seed for reproducibility.
    dist : {"uniform", "beta"}
        Distribution for noise. "uniform" -> U(0,1). "beta" -> Beta(a, b) in [0,1].
    beta_a, beta_b : float
        Parameters for Beta distribution when dist="beta".
    output_dtype : np.dtype
        Dtype of returned observations (default float32).
    """

    def __init__(
        self,
        env: gym.Env,
        noisy_indices: Iterable[int],
        seed: Optional[int] = None,
        dist: str = "uniform",
        beta_a: float = 2.0,
        beta_b: float = 2.0,
        output_dtype: np.dtype = np.float32,
    ):
        super().__init__(env)

        if not isinstance(self.observation_space, Box):
            raise TypeError("NoisyMetricAugmentWrapper requires a Box observation space.")

        if self.observation_space.shape is None or len(self.observation_space.shape) != 1:
            raise ValueError(
                f"Expected 1-D observations, got shape {self.observation_space.shape}."
            )

        self.base_dim: int = int(self.observation_space.shape[0])

        # Normalize and validate indices
        self.noisy_indices: List[int] = sorted(int(i) for i in noisy_indices)
        if len(set(self.noisy_indices)) != len(self.noisy_indices):
            raise ValueError("noisy_indices must be unique.")
        if len(self.noisy_indices) == 0:
            raise ValueError("Provide at least one noisy index.")

        # Final length = base_dim + #noise (we interleave noise, not replace)
        self.total_len: int = self.base_dim + len(self.noisy_indices)

        if min(self.noisy_indices) < 0 or max(self.noisy_indices) >= self.total_len:
            raise ValueError(
                f"noisy_indices must be in [0, {self.total_len - 1}] for final length {self.total_len}."
            )

        # Build a mapping from final index -> ('noise') or (base_idx)
        # Non-noisy slots are filled by base features in order.
        self.final_is_noise = np.zeros(self.total_len, dtype=bool)
        self.final_is_noise[self.noisy_indices] = True

        self.final_to_base: List[Optional[int]] = [None] * self.total_len
        b = 0
        for j in range(self.total_len):
            if not self.final_is_noise[j]:
                if b >= self.base_dim:
                    raise RuntimeError(
                        "Internal error: more non-noisy slots than base features."
                    )
                self.final_to_base[j] = b
                b += 1
        if b != self.base_dim:
            raise RuntimeError(
                "Internal error: number of non-noisy slots does not equal base_dim."
            )

        # Build the new observation space bounds element-wise
        base_low = np.array(self.observation_space.low, dtype=np.float32)
        base_high = np.array(self.observation_space.high, dtype=np.float32)

        low = np.empty(self.total_len, dtype=np.float32)
        high = np.empty(self.total_len, dtype=np.float32)
        for j in range(self.total_len):
            if self.final_is_noise[j]:
                low[j], high[j] = 0.0, 1.0
            else:
                bi = self.final_to_base[j]
                low[j], high[j] = base_low[bi], base_high[bi]

        self.observation_space = Box(low=low, high=high, dtype=output_dtype)

        # RNG & noise config
        self.rng = np.random.default_rng(seed)
        self.dist = dist.lower()
        if self.dist not in ("uniform", "beta"):
            raise ValueError("dist must be 'uniform' or 'beta'.")
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        self.output_dtype = output_dtype

    # ------------- Gym/Gymnasium API -------------

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 1 or obs.shape[0] != self.base_dim:
            raise ValueError(
                f"Expected base obs of shape ({self.base_dim},), got {obs.shape}."
            )

        out = np.empty(self.total_len, dtype=self.output_dtype)

        # Fill noise
        if self.dist == "uniform":
            out[self.final_is_noise] = self.rng.random(np.count_nonzero(self.final_is_noise))
        else:  # beta
            out[self.final_is_noise] = self.rng.beta(
                self.beta_a, self.beta_b, size=np.count_nonzero(self.final_is_noise)
            )

        # Fill base features in the remaining slots (preserve order)
        for j in range(self.total_len):
            bi = self.final_to_base[j]
            if bi is not None:
                out[j] = obs[bi]

        return out.astype(self.output_dtype, copy=False)

    # Optional: expose a helper for debugging/inspection
    def mapping_summary(self) -> Tuple[np.ndarray, List[Optional[int]]]:
        """
        Returns:
            final_is_noise: boolean mask over final indices
            final_to_base: mapping from final index -> base index (or None if noise)
        """
        return self.final_is_noise.copy(), list(self.final_to_base)


import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ThroughputCallback(BaseCallback):
    def __init__(self, ema=0.9, verbose=0):
        super().__init__(verbose)
        self.ema = ema
        self._last_t = None
        self._last_steps = None
        self._ema_sps = None

    def _on_training_start(self) -> None:
        self._last_t = time.time()
        self._last_steps = self.model.num_timesteps
        self._ema_sps = None

    def _on_rollout_end(self) -> None:
        now = time.time()
        steps = self.model.num_timesteps
        d_steps = steps - self._last_steps
        d_time = max(1e-6, now - self._last_t)
        sps = d_steps / d_time
        self._ema_sps = sps if self._ema_sps is None else self.ema*self._ema_sps + (1-self.ema)*sps

        # do TensorBoard
        self.logger.record("perf/steps_per_sec", float(sps))
        self.logger.record("perf/steps_per_sec_ema", float(self._ema_sps))

        # reset okna
        self._last_t = now
        self._last_steps = steps
    
    def _on_step(self) -> bool:
        # Required by BaseCallback; return True to continue training
        return True

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SmallCnnFor15xW(BaseFeaturesExtractor):
    """
    Safe for small widths (W=3 or 7). Preserves width with padding or (2,1) kernels,
    shrinks height only, then flattens → Linear(out_dim).
    """
    def __init__(self, observation_space, out_dim: int = 128):
        super().__init__(observation_space, features_dim=out_dim)
        c, h, w = observation_space.shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),   # SAME: (15, W)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(2,1), stride=1, padding=0),  # shrink H→14, keep W
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2,1), padding=(1,0)),  # H halves (≈7), keep W
            nn.ReLU(inplace=True),
        )
        # Infer flatten size from a dummy pass (avoids hand-counting for W=3 vs 7)
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flat = self.conv(sample).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x












import os
import numpy as np
from typing import List, Optional, Set

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback that:
      1) Saves the *current* model whenever the rolling mean reward (last 100 eps) improves.
      2) Saves intermediate snapshots evenly spaced across training (optional).
      3) Saves at predefined absolute timesteps `milestones` (e.g., [50_000, 100_000, 500_000]).

    Args:
        check_freq: how often (in env steps) to check rewards for "best model" saving.
        log_dir: directory with Monitor logs (for reward extraction) and where models are saved.
        total_timesteps: planned total steps (used only for intermediate spacing).
        verbose: verbosity.
        intermediate_models: number of evenly spaced intermediate snapshots (0 to disable).
        milestones: list of absolute timesteps at which to save a snapshot exactly once.
                    Example: [50_000, 100_000, 500_000]
        model_prefix: filename prefix for saved models (default: 'model')
    """

    def __init__(
        self,
        check_freq: int,
        log_dir: str,
        total_timesteps: int,
        verbose: int = 1,
        intermediate_models: int = 5,
        milestones: Optional[List[int]] = None,
        model_prefix: str = "model",
    ):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.log_dir = log_dir
        self.total_timesteps = int(total_timesteps)
        self.intermediate_models = int(intermediate_models)
        self.model_prefix = str(model_prefix)

        # best model path
        self.best_path = os.path.join(self.log_dir, "best_model")

        # internal state
        self.best_mean_reward = -np.inf
        self.save_count = 0  # for intermediate snapshots

        # milestones
        self.milestones = sorted(set(int(m) for m in (milestones or [])))
        self._pending_milestones: Set[int] = set(self.milestones)  # not yet saved

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        # We create folder for best model too (SB3 will create parent dirs if needed)
        os.makedirs(self.best_path, exist_ok=True) if self.best_path.endswith(os.sep) else os.makedirs(
            os.path.dirname(self.best_path), exist_ok=True
        )

        if self.verbose and self.milestones:
            print(f"[SaveCB] Milestones: {self.milestones}")

    # ---------- helpers ----------

    def _save_intermediate_if_due(self) -> None:
        """
        Save evenly spaced snapshots across training.
        """
        if self.intermediate_models <= 0 or self.total_timesteps <= 0:
            return

        freq = max(1, self.total_timesteps // self.intermediate_models)
        # save at steps: freq, 2*freq, ..., intermediate_models*freq ~= total_timesteps
        if self.num_timesteps > 0 and (self.num_timesteps % freq == 0):
            path = os.path.join(
                self.log_dir, f"{self.model_prefix}_intermediate_{self.save_count:02d}_{self.num_timesteps}"
            )
            if self.verbose > 0:
                print(f"[SaveCB] Saving intermediate #{self.save_count} at {self.num_timesteps} -> {path}.zip")
            self.model.save(path)
            self.save_count += 1

    def _save_milestones_if_due(self) -> None:
        """
        Save at the first time we reach or pass each milestone.
        """
        if not self._pending_milestones:
            return

        # milestones can be hit out of order if resuming training;
        # save any that are <= current timesteps and still pending
        to_save = [m for m in self._pending_milestones if self.num_timesteps >= m]
        for m in sorted(to_save):
            path = os.path.join(self.log_dir, f"{self.model_prefix}_step_{m}")
            if self.verbose > 0:
                print(f"[SaveCB] Saving milestone at {m} steps -> {path}.zip")
            self.model.save(path)
            self._pending_milestones.remove(m)

    def _save_best_if_improved(self) -> None:
        """
        Check rolling mean reward and save best model if improved.
        """
        # only check at the chosen frequency to avoid constant disk I/O
        if self.check_freq <= 0 or (self.num_timesteps % self.check_freq != 0):
            return

        # Retrieve training reward curve from Monitor logs
        try:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
        except Exception:
            # If no monitor files yet (or broken), skip silently
            return

        if len(x) == 0:
            return

        mean_reward = float(np.mean(y[-100:]))
        if self.verbose > 0:
            print(f"[SaveCB] t={self.num_timesteps}  best_mean={self.best_mean_reward:.2f}  "
                  f"last100_mean={mean_reward:.2f}")

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"[SaveCB] New best mean reward ({mean_reward:.2f}). Saving -> {self.best_path}.zip")
            self.model.save(self.best_path)

    # ---------- SB3 hook ----------

    def _on_step(self) -> bool:
        # A) best model by reward
        self._save_best_if_improved()
        # B) evenly spaced intermediate checkpoints
        self._save_intermediate_if_due()
        # C) predefined milestone checkpoints
        self._save_milestones_if_due()
        return True
