import sys
import os
import re
import math
import os.path as osp
import gym
import gym_cloudsimplus
import tensorflow as tf
import numpy as np
import json
import requests
import base64
import datetime
import time

from collections import defaultdict
from driver.common.vec_env import VecEnv
from driver.common.cmd_util import (
    common_arg_parser,
    parse_unknown_args,
    make_vec_env,
)
from driver.common.tf_util import get_session
from driver.common.swf_jobs import get_jobs_from_file
from driver.common.db_jobs import (
    get_cores_count_between,
    get_jobs_between,
    init_dbs,
)
from driver import logger
from importlib import import_module


oracle_update_url = os.getenv("ORACLE_UPDATE_URL",
                              "http://oracle:8080/update")


test_workload = [
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


def train(args,
          extra_args,
          old_env=None,
          old_model=None,
          ):

    env_type, env_id = get_env_type(args)

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args, extra_args)

    if env:
        logger.info('New environment built...')

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
            old_model=old_model,
            **alg_kwargs
        )
    else:
        logger.info('No new environment built - skipping training')
        env = old_env
        model = old_model

    return model, env


def adjust_submission_delay(job, start_timestamp):
    job['submissionDelay'] -= start_timestamp
    return job


def get_workload(args, extra_args):
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
        jobs = get_jobs_from_file(fpath=args.workload_file,
                                  mips_per_core=args.mips_per_core)
        return jobs

    if args.continuous_mode:
        training_data_start = int(extra_args['training_data_start'])
        training_data_end = int(extra_args['training_data_end'])
        jobs = get_jobs_between(training_data_start, training_data_end)

        # the timestamp of the job should be a real timestamp from the
        # prod system - we need to move it to the beginning of the simulation
        # otherwise we will wait for eternity (almost)
        time_adjusted_jobs = [
            adjust_submission_delay(job, training_data_start)
            for job
            in jobs
        ]

        logger.info('DEBUG DEBUG DEBUG Jobs retrieved from database')
        for job in jobs:
            logger.info(f'Job: {job}')

        return time_adjusted_jobs

    raise RuntimeError('Please use the test, workload_file or '
                       'continuous mode options')


def build_env(args, extra_args):
    alg = args.alg
    seed = args.seed
    initial_vm_count = args.initial_vm_count
    simulator_speedup = args.simulator_speedup

    workload = get_workload(args, extra_args)
    if len(workload) == 0:
        logger.info('Cannot create a working environment without any jobs...')
        return None

    env_type, env_id = get_env_type(args)
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    initial_s_vm_count = extra_args.get('initial_s_vm_count', initial_vm_count)
    initial_m_vm_count = extra_args.get('initial_m_vm_count', initial_vm_count)
    initial_l_vm_count = extra_args.get('initial_l_vm_count', initial_vm_count)

    # how many iterations are in the hour? * core iteration running cost * 2
    # x2 because a S vm has 2 cores
    s_vm_hourly_running_cost = args.core_iteration_cost * 2
    s_vm_hourly_running_cost *= (3600 / simulator_speedup)

    env_kwargs = {
        'initial_s_vm_count': initial_s_vm_count,
        'initial_m_vm_count': initial_m_vm_count,
        'initial_l_vm_count': initial_l_vm_count,
        'jobs_as_json': json.dumps(workload),
        'simulation_speedup': str(simulator_speedup),
        'split_large_jobs': 'true',
        'vm_hourly_running_cost': s_vm_hourly_running_cost,
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
        logger.configure(log_path, **kwargs)
    else:
        logger.configure(**kwargs)


def test_model(model, env):
    obs = env.reset()
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))
    episode_rew = 0
    done = False
    while not done:
        if state is not None:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        obs_arr = env.render(mode='array')
        obs_last = [serie[-1] for serie in obs_arr]
        #print(f'{obs_last} | rew: {rew} | ep_rew: {episode_rew}')

    return episode_rew


def calculate_reward(core_iteration_cost, cores_counts):
    total_reward = 0
    for metric_values in cores_counts:
        values = [point.value for point in metric_values]
        for value in values:
            total_reward -= value * core_iteration_cost
    return total_reward


def update_oracle_policy(model):
    path = 'updated_model.bin'
    model.save(path)

    with open(path, 'rb') as updated_model:
        updated_model_bytes = updated_model.read()
        updated_model_encoded = base64.b64encode(updated_model_bytes)

    req_payload = {
        "content": updated_model_encoded,
        "file_suffix": datetime.date.today().isoformat(),
        "network_type": "lstm",
    }

    resp = requests.post(oracle_update_url, data=req_payload)
    logger.log(f'Model updated ({resp.status_code}): {resp.text}')


def data_available(val):
    if val is not None:
        if len(val) > 0:
            return True

    return False


def training_loop(args, extra_args):
    logger.log('Initializing databases')
    init_dbs()
    logger.log('Initialized databases')
    logger.log('Training loop: starting...')

    base_save_path = osp.expanduser(args.save_path)
    logger.log(f'Training loop: saving models in: {base_save_path}')

    iteration_length_s = args.iteration_length_s
    # global training start
    global_start = time.time()
    initial_timestamp = global_start if args.initial_timestamp < 0 else args.initial_timestamp
    iterations = 0
    running = True
    model = None
    env = None

    date_tag = datetime.datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
    base_save_path = os.path.join(base_save_path, date_tag)

    while running:
        logger.debug(f'Training loop: iteration {iterations}')

        # start of the current iteration
        iteration_start = time.time()
        # how much time had passed since the start of the training
        iteration_delta = iteration_start - global_start

        current_tstamp_for_db = initial_timestamp + iteration_delta
        training_data_end = current_tstamp_for_db
        training_data_start = current_tstamp_for_db - iteration_length_s

        updated_extra_args = {
            'iteration_start': iteration_start,
            'training_data_start': training_data_start,
            'training_data_end': training_data_end,
        }
        updated_extra_args.update(extra_args)

        cores_count = get_cores_count_between(training_data_start,
                                              training_data_end)

        logger.log(f'Using training data starting at: '
                   f'{training_data_start}-{training_data_end} '
                   f'Initial tstamp: {initial_timestamp}')

        if all([data_available(v) for k, v in cores_count.items()]):
            logger.log(f'Initial cores: {cores_count}')
            updated_extra_args.update({
                'initial_s_vm_count': math.floor(cores_count['s_cores'][0]/2),
                'initial_m_vm_count': math.floor(cores_count['m_cores'][0]/2),
                'initial_l_vm_count': math.floor(cores_count['l_cores'][0]/2),
            })

            model, env = train(args,
                               updated_extra_args,
                               old_env=env,
                               old_model=model)

            if model:
                model_save_path = f'{base_save_path}_{iterations}.bin'
                model.save(model_save_path)
                new_policy_total_reward = test_model(model, env)
                old_policy_total_reward = calculate_reward(
                    args.core_iteration_cost,
                    cores_count)

                logger.info(f'Old policy reward: {old_policy_total_reward} '
                            f'new policy reward: {new_policy_total_reward}')
                if new_policy_total_reward > old_policy_total_reward:
                    logger.info('New policy has a higher reward, '
                                'updating the policy')
                    update_oracle_policy(model)
        else:
            logger.log(f'Cannot initialize vm counts - not enough data '
                       f'available. Cores count: {cores_count}')

        iteration_end = time.time()
        iteration_len = iteration_end - iteration_start
        time_fill = iteration_length_s - iteration_len
        if time_fill > 0:
            logger.log(f'Sleeping for {time_fill}')
            time.sleep(time_fill)
        iterations += 1

    env.close()
    logger.log('Training loop: ended')


def training_once(args, extra_args):
    logger.log('Training once: starting...')
    model, env = train(args, extra_args)

    if args.save_path is not None:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.log("Running trained model")
        test_model(model, env)

    env.close()
    logger.log('Training once: ended')


def main():
    args = sys.argv
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    configure_logger(args.log_path, format_strs=['stdout', 'log'])
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.continuous_mode:
        training_loop(args, extra_args)
    else:
        training_once(args, extra_args)


if __name__ == '__main__':
    main()
