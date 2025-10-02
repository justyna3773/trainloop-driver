"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenObservation, FilterObservation
from driver import logger
from driver.bench import Monitor
from driver.common import set_global_seeds
from driver.common.atari_wrappers import make_atari, wrap_deepmind
from driver.common.vec_env.subproc_vec_env import SubprocVecEnv
from driver.common.vec_env.dummy_vec_env import DummyVecEnv
from driver.common import retro_wrappers
from driver.common.wrappers import ClipActionsWrapper


def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=10.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()

    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer
        )

    if not force_dummy and num_env > 1:
        return SubprocVecEnv([
            make_thunk(i + start_index, initializer=initializer)
            for i in range(num_env)
        ])
    else:
        return DummyVecEnv([
            make_thunk(i + start_index, initializer=None)
            for i in range(num_env)
        ])


def make_env(env_id,
             env_type,
             mpi_rank=0,
             subrank=0,
             reward_scale=10.0,
             gamestate=None,
             flatten_dict_observations=True,
             wrapper_kwargs=None,
             env_kwargs=None,
             logger_dir=None,
             initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*', '', env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    if env_type == 'atari':
        env = make_atari(env_id)
    elif env_type == 'retro':
        import retro
        gamestate = gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(
            game=env_id,
            max_episode_steps=10000,
            use_restricted_actions=retro.Actions.DISCRETE,
            state=gamestate)
    else:
        env = gym.make(env_id, **env_kwargs)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    path = os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank))
    env = Monitor(env,
                  logger_dir and path,
                  allow_early_resets=True)

    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    elif env_type == 'retro':
        if 'frame_stack' not in wrapper_kwargs:
            wrapper_kwargs['frame_stack'] = 1
        env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)

    return env


def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed  + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from driver.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env

def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def atari_arg_parser():
    """
    Create an argparse.ArgumentParser for run_atari.py.
    """
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def mujoco_arg_parser():
    print('Obsolete - use common_arg_parser instead')
    return common_arg_parser()

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default="ThreeSizeAppEnv-v1")
                        #default='SingleDCAppEnv-v0')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=str, default=42)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1000)#default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--initial_vm_count', type=int, default=1)
    parser.add_argument('--simulator_speedup', type=float, default=600.0) # with simulator speedup 1000 the task becomes too easy
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--workload_file', default="./TEST-DNNEVO-2.swf", type=str)
    parser.add_argument('--mips_per_core', default=5000, type=int)#higher mips_per_core, easier task theoretically
    parser.add_argument('--continuous_mode', default=False, action='store_true')
    parser.add_argument('--evaluate_continuous_mode', default=False, action='store_true')
    parser.add_argument('--evaluate_mode', default=False, action='store_true')
    parser.add_argument('--iteration_length_s', default=300, type=int,)
    parser.add_argument('--core_iteration_cost', default=0.00166666666, type=float)
    parser.add_argument('--initial_timestamp', default=-1, type=int)
    parser.add_argument('--initial_model', default=None, type=str)
    parser.add_argument('--initial_vm_count_no_data', default=None, type=int)
    parser.add_argument('--reindex_jobs', default=True, type=bool)
    parser.add_argument('--algo', help='Algorithm', type=str, default='PPO')
    parser.add_argument('--policy', help='Algorithm', type=str, default='MlpPolicy')
    parser.add_argument('--observation_history_length', help='observation_history_length', type=str, default='1')
    parser.add_argument('--queue_wait_penalty', type=float, default=0.0005)
                        #default=0.00001)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--continue_training', default=False)

    return parser

def robotics_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    return parser


def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
