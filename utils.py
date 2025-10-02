FEATURE_NAMES = [ "vmAllocatedRatio",
            "avgCPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtilization",
            "p90CPUUtilization",
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

def test_model_lstm(model, env, n_runs=6, deterministic=True):
    """
    Eval for SB3 policies (MlpPolicy, MlpLstmPolicy, etc.) on (Vec)Env.
    Assumes a single-env VecEnv (num_envs==1) or a raw Gym env.
    Returns the same tuple shape as your transformer test_model.
    """
    is_vec = _is_vec_env(env)
    if is_vec and getattr(env, "num_envs", 1) != 1:
        raise ValueError(f"Evaluate with a single-env VecEnv. Got num_envs={env.num_envs}.")

    # If using VecNormalize, disable updates and de-normalize rewards
    if hasattr(env, "training"):
        env.training = False
    if hasattr(env, "norm_reward"):
        env.norm_reward = False

    all_obs_7 = []                 # we will try to extract a (7,) if possible; else store None
    all_actions = []
    ep_rewards_per_run, ep_lengths = [], []
    ep_rewards_list, ep_obs_per_run_7 = [], []

    state = None
    for _ in range(n_runs):
        if is_vec:
            obs = env.reset()
            episode_start = np.array([True], dtype=bool)
        else:
            obs = env.reset()
            episode_start = None

        ep_rew, ep_len = 0.0, 0
        rewards_run, obs_run_7 = [], []

        done = False
        while not done:
            # SB3 recurrent-friendly predict
            try:
                action, state = model.predict(
                    obs, state=state, episode_start=episode_start, deterministic=deterministic
                )
            except TypeError:
                # Older SB3 signature
                action, state = model.predict(obs, state=state, deterministic=deterministic)

            if is_vec:
                obs, rews, dones, infos = env.step(action)
                rew = float(rews[0])
                done = bool(dones[0])
                episode_start = dones
            else:
                obs, rew, done, info = env.step(action)
                rew = float(rew)

            # Try to extract a (7,) vector if obs is a Dict with 'h'
            if isinstance(obs, dict) and "h" in obs:
                h = np.asarray(obs["h"], dtype=np.float32)
                h = h[0] if (h.ndim == 2 and h.shape[0] == 1) else h
                if h.shape == (7,):
                    all_obs_7.append(h)
                    obs_run_7.append(h)
                else:
                    all_obs_7.append(None)
            else:
                all_obs_7.append(None)

            all_actions.append(action)
            ep_rew += rew
            rewards_run.append(rew)
            ep_len += 1

            if done:
                break

        ep_rewards_per_run.append(ep_rew)
        ep_lengths.append(ep_len)
        ep_rewards_list.append(np.asarray(rewards_run, dtype=np.float32))
        ep_obs_per_run_7.append(np.stack(obs_run_7, axis=0) if len(obs_run_7) > 0 else np.zeros((0,7), np.float32))

        state = None
        if is_vec:
            episode_start = np.array([True], dtype=bool)

    mean_episode_reward = float(np.mean(ep_rewards_per_run)) if ep_rewards_per_run else 0.0
    print(f"Mean reward over {n_runs} runs: {mean_episode_reward:.6f}")

    # pack to match the transformer function’s return signature
    observations = np.asarray([x for x in all_obs_7 if x is not None], dtype=np.float32)
    if observations.size > 0:
        observations = observations[:, None, :]   # (x, 1, 7)
    else:
        observations = np.zeros((0,1,7), dtype=np.float32)

    episode_lengths = np.asarray(ep_lengths, dtype=np.int32)
    episode_rewards = np.asarray(ep_rewards_list, dtype=object)
    episode_rewards_per_run = np.asarray(ep_rewards_per_run, dtype=np.float32)
    observations_per_run = np.asarray(ep_obs_per_run_7, dtype=object)
    actions = np.asarray(all_actions, dtype=object)

    return (mean_episode_reward, observations, episode_lengths,
            episode_rewards, episode_rewards_per_run, observations_per_run, actions)

#TODO this is basic evaluate function
def evaluate_backup(model, env, n_episodes=15, deterministic=False, freeze_vecnorm=True):
    # keep VecNormalize reward semantics = true env reward
    if hasattr(env, "norm_reward"): env.norm_reward = False
    if freeze_vecnorm and hasattr(env, "training"): env.training = False

    is_vec = hasattr(env, "num_envs")
    if is_vec and env.num_envs != 1:
        raise ValueError("Use a single-env VecEnv for eval.")

    state = None
    ep_returns = []
    observations = []
    actions = []

    for _ in range(n_episodes):
        obs = env.reset()
        episode_start = np.array([True], bool) if is_vec else None
        done = False; ret = 0.0 
        while not done:
            try:
                action, state = model.predict(obs, state=state,
                                              episode_start=episode_start,
                                              deterministic=deterministic)
            except TypeError:
                action, state = model.predict(obs, state=state,
                                              deterministic=deterministic)
            if is_vec:
                obs, r, d, info = env.step(action)
                observations.append(obs)
                actions.append(action)
                r = float(r[0]); done = bool(d[0]); episode_start = d
            else:
                obs, r, done, info = env.step(action)
                observations.append(obs)
                actions.append(action)
            ret += float(r)
        ep_returns.append(ret)
        state = None
    return float(np.mean(ep_returns)), observations, ep_returns, actions

#TODO last backup function that I was using
def evaluate_backup(model, env, n_episodes=15, deterministic=False,
             freeze_vecnorm=True, collect_info=True):
    # keep VecNormalize reward semantics = true env reward
    if hasattr(env, "norm_reward"): env.norm_reward = False
    if freeze_vecnorm and hasattr(env, "training"): env.training = False

    is_vec = hasattr(env, "num_envs")
    if is_vec and env.num_envs != 1:
        raise ValueError("Use a single-env VecEnv for eval.")

    state = None
    ep_returns, observations, actions = [], [], []
    raw_obs_trace, raw_metrics_trace = [], []  # <— new

    for _ in range(n_episodes):
        obs = env.reset()
        episode_start = np.array([True], bool) if is_vec else None
        done = False; ret = 0.0

        while not done:
            try:
                action, state = model.predict(
                    obs, state=state, episode_start=episode_start,
                    deterministic=deterministic)
            except TypeError:
                action, state = model.predict(
                    obs, state=state, deterministic=deterministic)

            if is_vec:
                obs, r, d, info = env.step(action)
                observations.append(obs); actions.append(action)
                r = float(r[0]); done = bool(d[0]); episode_start = d
                if collect_info:
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    raw_obs_trace.append(i0.get("_raw_obs"))
                    raw_metrics_trace.append(i0.get("_raw_metrics"))
            else:
                obs, r, done, info = env.step(action)
                observations.append(obs); actions.append(action)
                if collect_info:
                    raw_obs_trace.append(info.get("_raw_obs"))
                    raw_metrics_trace.append(info.get("_raw_metrics"))

            ret += float(r)

        ep_returns.append(ret)
        state = None

    mean_return = float(np.mean(ep_returns)) if ep_returns else 0.0
    if collect_info:
        return mean_return, raw_obs_trace, ep_returns, actions
    else:
        return mean_return, observations, ep_returns, actions
import numpy as np

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
                #model = RecurrentPPO.load(r'C:\initial_model\historic_synthetic_dnnevo\RecurrentPPO_19_worst\recurrentppo_MlpLstmPolicy_mlplstm_policy_worst', env=env)
                model = RecurrentPPO.load(path=rf'C:\initial_model\{algo.lower()}\{policy}\{algo.lower()}_{policy}')
                #                                     env=env)
            elif PCA:
                pass   
            else:
                model = PPO.load(path=rf'C:\initial_model\{algo.lower()}\{policy}\{model_name}',env=env)
                #model = PPO.load(path=rf'C:\Users\ultramarine\Desktop\ppo_magisterka\trainloop_driver_official\trainloop_driver_final\trainloop-driver\output_malota\ppo_mlp_intermediate_02_300000',env=env)
                #model = PPO.load(path=r'C:\Users\ultramarine\Desktop\ppo_magisterka\trainloop_driver_official\trainloop_driver_final\trainloop-driver\output_malota\ppo_gawrl_intermediate_02_300000', env=env)
                #model = PPO.load(r'C:\initial_model\historic_synthetic_dnnevo\PPO_43_baseline\ppo_MlpPolicy_mlp_vm_waiting', env=env)
                
            env.reset()
            #mean_reward, observations, episode_lenghts, rewards, rewards_per_run, observations_evalute_results, actions  = test_model(model, env, n_runs=10)
            if PCA_OFFLINE:
                mean_reward, observations = evaluate_pca(model, env)
            else:
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

#TODO this is extended evaluate function for PCA_OFFLINE
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecNormalize
def _freeze_vecnormalize_anywhere(env, enable=True):
    if not enable: return
    cur = env
    seen = set()
    # unwrap VecEnv wrappers
    while isinstance(cur, VecEnvWrapper) and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, VecNormalize):
            cur.training = False
            cur.norm_reward = False   # ensure TRUE rewards at eval
        cur = cur.venv
    # handle base case
    if isinstance(cur, VecNormalize):
        cur.training = False
        cur.norm_reward = False
def evaluate_pca(model, env, n_episodes=15, deterministic=True, freeze_vecnorm=True, collect_raw=True):
    _freeze_vecnormalize_anywhere(env, enable=freeze_vecnorm)
    is_vec = isinstance(env, VecEnv) or hasattr(env, "num_envs")
    if is_vec and getattr(env, "num_envs", 1) != 1:
        raise ValueError("Use a single-env VecEnv (num_envs=1) for eval.")

    state = None
    ep_returns = []
    observations = []  # will store raw pre-PCA if collect_raw=True else projected obs

    for _ in range(n_episodes):
        obs = env.reset()  # projected obs (PCA) if PCA wrapper is active

        # try to grab the initial *raw* obs saved by the PCA wrapper on reset
        if collect_raw:
            try:
                # VecEnv path (PCAVecObsWrapper): attribute lives on the wrapper; get_attr returns [value]
                initial_raw = env.get_attr("_last_raw_obs")[0] if is_vec else getattr(env, "_last_raw_obs", None)
                if initial_raw is not None:
                    observations.append(np.asarray(initial_raw).copy())
            except Exception:
                pass  # if not present, we’ll just start collecting from first step

        episode_start = np.array([True], dtype=bool) if is_vec else None
        done = False
        ret = 0.0

        while not done:
            # always pass the PCA/projected obs to the model
            try:
                action, state = model.predict(
                    obs, state=state, episode_start=episode_start, deterministic=deterministic
                )
            except TypeError:
                action, state = model.predict(obs, state=state, deterministic=deterministic)

            step_out = env.step(action)

            # Gymnasium 5-tuple vs Gym 4-tuple handling
            if len(step_out) == 5:
                obs_next, r, terminated, truncated, info = step_out
                d = terminated or truncated
            else:
                obs_next, r, d, info = step_out

            # Collect raw obs without disturbing what the model sees next
            if collect_raw:
                if is_vec:
                    # SB3 VecEnv: info is a list of dicts
                    raw = info[0].get("raw_obs") if isinstance(info, (list, tuple)) and info else None
                else:
                    raw = info.get("raw_obs") if isinstance(info, dict) else None
                observations.append(np.asarray(raw).copy() if raw is not None else np.asarray(obs_next).copy())
            else:
                observations.append(np.asarray(obs_next).copy())

            # Prepare for next step
            if is_vec:
                r_val = float(np.asarray(r).reshape(-1)[0])
                d_val = bool(np.asarray(d).reshape(-1)[0])
                episode_start = np.asarray([d_val], dtype=bool)
            else:
                r_val = float(r)
                d_val = bool(d)

            ret += r_val
            done = d_val
            obs = obs_next  # keep feeding PCA/projected obs to the model

        ep_returns.append(ret)
        state = None

    return float(np.mean(ep_returns)), observations


def run_episode(env, policy_fn):
    obs = env.reset()
    ep_ret = 0.0
    while True:
        a = policy_fn(obs)  # e.g., lambda o: ACTION_NOTHING
        obs, r, done, info = env.step([a])
        ep_ret += float(r)
        if done:
            return ep_ret


import numpy as np
import gym
from gym import spaces
# utils.py
import gym
import numpy as np

#TODO last correct working FeatureMaskWrapper
class FeatureMaskWrapper_backup(gym.Wrapper):
    """
    Masks observations for the agent, but exposes the original, unmasked values
    via `info["_raw_obs"]` (and optionally `info["_raw_metrics"]`) on step().
    Set expose_raw=True only for eval to keep training light.
    """
    def __init__(self, env, mask, expose_raw=False, raw_metrics_fn=None):
        super().__init__(env)
        self.mask = np.asarray(mask, dtype=bool)
        self.expose_raw = expose_raw
        self.raw_metrics_fn = raw_metrics_fn  # callable: raw_obs -> dict

    def _apply_mask(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        masked = obs.copy()
        masked[~self.mask] = 0.0
        return masked

    # Old Gym API (as used in your code): reset() returns obs only
    # If you later switch to Gymnasium, add the (obs, info) path.
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._apply_mask(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        raw_obs = np.asarray(obs, dtype=np.float32)
        masked_obs = self._apply_mask(raw_obs)

        if self.expose_raw:
            # make sure we can attach fields even if info is a specialized mapping
            info = dict(info) if info is not None else {}
            info["_raw_obs"] = raw_obs
            if self.raw_metrics_fn is not None:
                # compute whatever "original metrics" you want from the raw obs
                info["_raw_metrics"] = self.raw_metrics_fn(raw_obs)

        return masked_obs, reward, done, info

# FeatureMaskWrapper: zero out features along the LAST dimension, keep shape.
# Example: obs (1, 15, 7) + mask [1,1,0,0,0,0,1] -> output still (1, 15, 7),
# but features 2..5 are always zero.

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

from typing import Iterable, Optional, Sequence
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class SelectMetricsWrapper_backup(gym.ObservationWrapper):
    """
    Keep only metrics at selected indices (or mask) along the LAST axis.
    Examples:
      - obs (1, 15, 7) + mask [1,1,0,0,0,0,1] -> (1, 15, 3)
      - obs (1, 8, 20)  + indices [0,3,7]     -> (1, 8, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        metric_idxs: Optional[Iterable[int]] = None,
        mask: Optional[Sequence[int]] = None,
        output_dtype: Optional[np.dtype] = None,
    ):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("SelectMetricsWrapper supports only Box observation spaces.")
        if env.observation_space.shape is None or len(env.observation_space.shape) < 1:
            raise ValueError("Observation space must be at least 1-D.")

        self._pre_shape = tuple(env.observation_space.shape)
        self._feat_dim = int(self._pre_shape[-1])

        # Build selected indices from mask or metric_idxs
        if metric_idxs is not None and mask is not None:
            raise ValueError("Provide either 'metric_idxs' or 'mask', not both.")

        if mask is not None:
            mask = np.asarray(mask, dtype=int).ravel()
            if mask.size != self._feat_dim:
                raise ValueError(f"Mask length {mask.size} != feature dim {self._feat_dim}.")
            sel = np.flatnonzero(mask != 0)
        elif metric_idxs is not None:
            sel = np.unique(np.asarray(list(metric_idxs), dtype=np.int64))
        else:
            raise ValueError("You must provide 'metric_idxs' or 'mask'.")

        if sel.size == 0:
            raise ValueError("No features selected.")
        if sel.min() < 0 or sel.max() >= self._feat_dim:
            raise IndexError(f"indices must be in [0, {self._feat_dim-1}]")

        self.metric_idxs = sel.astype(np.int64)
        self.output_dtype = output_dtype or env.observation_space.dtype

        # Prepare low/high; handle scalar bounds by broadcasting to obs shape
        low = env.observation_space.low
        high = env.observation_space.high

        if np.isscalar(low) and np.isscalar(high):
            low_arr = np.full(self._pre_shape, low, dtype=np.float32)
            high_arr = np.full(self._pre_shape, high, dtype=np.float32)
        else:
            low_arr = np.asarray(low, dtype=np.float32)
            high_arr = np.asarray(high, dtype=np.float32)

        # Slice bounds along the last axis
        new_low = np.take(low_arr, self.metric_idxs, axis=-1)
        new_high = np.take(high_arr, self.metric_idxs, axis=-1)
        self._post_shape = tuple(self._pre_shape[:-1] + (self.metric_idxs.size,))

        # Sanity check shapes
        assert tuple(new_low.shape) == self._post_shape and tuple(new_high.shape) == self._post_shape, \
            f"Bound shapes mismatch: got {new_low.shape}, {new_high.shape}, expected {self._post_shape}"

        self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=self.output_dtype)

    @property
    def selected_indices(self):
        return self.metric_idxs.tolist()

    def observation(self, obs):
        # Slice runtime observation along the last axis
        out = np.take(np.asarray(obs), self.metric_idxs, axis=-1)
        return out.astype(self.output_dtype, copy=False)

# NoisyMetricAugmentWrapper
# Inserts per-step random features in [0,1] at user-chosen indices of the FINAL vector.
# Base features keep their original order and occupy the remaining indices.

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
