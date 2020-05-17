import sys
import re
import os.path as osp
import gym
import gym_cloudsimplus
from collections import defaultdict
import tensorflow as tf
import numpy as np
import json


from driver.common.vec_env import VecEnv
from driver.common.cmd_util import (
    common_arg_parser,
    parse_unknown_args,
    make_vec_env,
)
from driver.common.tf_util import get_session
from driver.common.swf_jobs import read_jobs_from_file
from driver import logger
from importlib import import_module

test_workload = [
    {
        'jobId': 1,
        'submissionDelay': 10,
        'mi': 500000,
        'numberOfCores': 1,
    }
]

test_workload2 = [
    {
        "jobId": 1441,
        "submissionDelay": 507,
        "mi": 30720000,
        "numberOfCores": 4096
    },
    {
        "jobId": 1442,
        "submissionDelay": 4775,
        "mi": 108400000,
        "numberOfCores": 8672
    },
    {
        "jobId": 1443,
        "submissionDelay": 5327,
        "mi": 470000000,
        "numberOfCores": 8000
    },
    {
        "jobId": 1445,
        "submissionDelay": 5653,
        "mi": 1097870000,
        "numberOfCores": 8696
    },
    {
        "jobId": 1447,
        "submissionDelay": 7514,
        "mi": 350000000,
        "numberOfCores": 8000
    },
    {
        "jobId": 1448,
        "submissionDelay": 7796,
        "mi": 180000000,
        "numberOfCores": 8000
    },
    {
        "jobId": 1449,
        "submissionDelay": 7882,
        "mi": 108400000,
        "numberOfCores": 8672
    },
    {
        "jobId": 1450,
        "submissionDelay": 7951,
        "mi": 401080000,
        "numberOfCores": 8672
    }
]


_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


def train(args, extra_args):
    env_type, env_id = get_env_type(args)

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    logger.info('Environment built...')

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    logger.info('Training {} on {}:{} with arguments \n{}'.format(
        args.alg,
        env_type,
        env_id,
        alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def get_workload(args):
    if args.test:
        logger.info('*** TEST MODE ***')
        logger.info('Using test workload ...')
        return test_workload

    if args.workload_file is not None:
        if args.mips_per_core is None:
            raise RuntimeError('You need to specify the number of MIPS per '
                               'core of the cluster used to generate the '
                               'loaded workload')

        logger.info(f'Loading workload from file: {args.workload_file}')
        jobs = read_jobs_from_file(fpath=args.workload_file,
                                   mips_per_core=args.mips_per_core)
        return jobs

    raise RuntimeError('DB access not implemented yet!')


def build_env(args):
    alg = args.alg
    seed = args.seed
    initial_vm_count = args.initial_vm_count
    simulator_speedup = args.simulator_speedup

    workload = get_workload(args)

    env_type, env_id = get_env_type(args)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    env_kwargs = {
        'initial_vm_count': str(initial_vm_count),
        'jobs_as_json': json.dumps(workload),
        'simulation_speedup': str(simulator_speedup),
    }

    flatten_dict_observations = alg not in {'her'}
    env = make_vec_env(env_id,
                       env_type,
                       args.num_env or 1,
                       seed,
                       reward_scale=args.reward_scale,
                       flatten_dict_observations=flatten_dict_observations,
                       env_kwargs=env_kwargs,
                       )

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from driver
        alg_module = import_module('.'.join(['driver', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary,
    evaluating python objects when possible
    '''
    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main():
    args = sys.argv
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    configure_logger(args.log_path)
    model, env = train(args, extra_args)

    if args.save_path is not None:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()
    return model


if __name__ == '__main__':
    main()
