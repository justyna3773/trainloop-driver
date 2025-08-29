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

def evaluate(model, env, n_episodes=15, deterministic=False,
             freeze_vecnorm=True, collect_info=False):
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


#TODO this is basic evaluate function
def evaluate_sample(args, extra_args):
    # build env
    from run import build_env
    import pandas as pd
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    
    env = build_env(args, extra_args)
    
    
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
                model = RecurrentPPO.load(path=rf'C:\initial_model\{algo.lower()}\{policy}\{algo.lower()}_{policy}',
                                                    env=env)
            elif PCA:
                pass   
            else:
                model = PPO.load(path=rf'C:\initial_model\{algo.lower()}\{policy}\{model_name}',
                                                    env=env)
                
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

class FeatureMaskWrapper(gym.Wrapper):
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
