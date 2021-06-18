import sys
import os
import re
import math
import os.path as osp
import gym
import gym_cloudsimplus
import numpy as np
import json
import requests
import base64
import datetime
import time
import pandas as pd

from collections import defaultdict
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DQN
from driver.common.cmd_util import (
    common_arg_parser,
    parse_unknown_args
)
from driver.common.swf_jobs import get_jobs_from_file
from driver.common.db import (
    get_cores_count_at,
    get_jobs_between,
    init_dbs,
)
from driver import logger
from importlib import import_module


submitted_jobs_cnt: int = 0
oracle_update_url: str = os.getenv("ORACLE_UPDATE_URL",
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

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from tqdm.auto import tqdm


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


def train(args,
          extra_args,
          env,
          ):

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    model = DQN('MlpPolicy', env)

    logger.info(model)
    with ProgressBarManager(total_timesteps) as pbar_callback:
        model.learn(total_timesteps, callback=pbar_callback)
    # learn = get_learn_function(args.alg)
    # model = learn(
    #     env=env,
    #     seed=seed,
    #     total_timesteps=total_timesteps,
    #     **alg_kwargs
    # )

    return model


def adjust_submission_delay(job, start_timestamp):
    job['submissionDelay'] -= start_timestamp
    return job


def ensure_mi_is_positive(jobs):
    for job in jobs:
        if job['mi'] < 1:
            logger.info(f'Job: {job} has mi < 1. Changing to 1')
            job['mi'] = 1


def get_workload(args, extra_args):
    global submitted_jobs_cnt

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

        ensure_mi_is_positive(jobs)

        if args.reindex_jobs:
            logger.info('Reindexing the jobs enabled - changing the ids.')
            for job in jobs:
                submitted_jobs_cnt += 1
                job['jobId'] = submitted_jobs_cnt

            submitted_jobs_cnt += 1
            extra_job_id = submitted_jobs_cnt

        else:
            max_id = 0
            for job in jobs:
                if job['jobId'] > max_id:
                    max_id = job['jobId']
            extra_job_id = max_id + 1

        # we want to always have at least one job at the end of the simulation
        if len(jobs) == 0:
            jobs.append({
                # id is the nanosecond of the last possible job submission + 1
                # which should give us enough room to avoid conflicts in ids
                'jobId': extra_job_id,
                'submissionDelay': training_data_end,
                'mi': 1,
                'numberOfCores': 1,
            })

        # the timestamp of the job should be a real timestamp from the
        # prod system - we need to move it to the beginning of the simulation
        # otherwise we will wait for eternity (almost)
        time_adjusted_jobs = [
            adjust_submission_delay(job, training_data_start)
            for job
            in jobs
        ]

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
        logger.error('This should never happen! '
                     'Cannot create a working environment without any jobs...')
        return None

    logger.info('Dumping jobs available for training')
    for job in workload:
        logger.info(f'{job}')

    env_type, env_id = get_env_type(args)
    logger.info(f'Env type: {env_type}, env id: {env_id}')
    # config = tf.ConfigProto(allow_soft_placement=True,
    #                         intra_op_parallelism_threads=1,
    #                         inter_op_parallelism_threads=1)
    # config.gpu_options.allow_growth = True
    # get_session(config=config, force_recreate=True)

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

    logger.info(args)
    env = make_vec_env(env_id=env_id,
                       n_envs=args.num_env or 1,
                       # seed=seed,
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
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(
            env_id, _game_envs.keys())

    return env_type, env_id


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
        print(obs)
        if state is not None:
            actions, _, state, _ = model.predict(obs, S=state, M=dones)
        else:
            actions, _states = model.predict(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        obs_arr = env.render(mode='array')
        obs_last = [serie[-1] for serie in obs_arr]
        print(f'{obs_last} | rew: {rew} | ep_rew: {episode_rew}')

    if isinstance(episode_rew, list):
        episode_rew = episode_rew[0]

    return episode_rew


def collect_observations(model, env):
    obs = env.reset()
    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))
    episode_rew = 0
    done = False
    while not done:
        print(obs)
        if state is not None:
            actions, _, state, _ = model.predict(obs, S=state, M=dones)
        else:
            actions, _states = model.predict(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        obs_arr = env.render(mode='array')
        obs_last = [serie[-1] for serie in obs_arr]


def update_oracle_policy(model_save_path):
    with open(model_save_path, 'rb') as updated_model:
        updated_model_bytes = updated_model.read()
        updated_model_encoded = base64.b64encode(updated_model_bytes)
        updated_model_str = updated_model_encoded.decode('utf-8')

    req_payload = {
        "content": updated_model_str,
        "file_suffix": datetime.datetime.now().isoformat(),
        "network_type": "lstm",
    }

    resp = requests.post(oracle_update_url, json=req_payload)
    logger.log(f'Model updated ({resp.status_code}): {resp.text}')


def data_available(val):
    if val is not None:
        return True

    return False


def training_loop(args, extra_args):
    logger.log('Initializing databases')
    init_dbs()
    logger.log('Initialized databases')
    logger.log('Training loop: initializing...')

    base_save_path = osp.expanduser(args.save_path)
    logger.log(f'Training loop: saving models in: {base_save_path}')

    iteration_length_s = args.iteration_length_s
    # global training start
    global_start = time.time()
    initial_timestamp = global_start if args.initial_timestamp < 0 else args.initial_timestamp
    iterations = 0
    running = True
    env = None

    date_tag = datetime.datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
    base_save_path = os.path.join(base_save_path, date_tag)

    best_model_path = '/best_model/best_model'
    best_model_replay_buffer_path = '/best_model/best_model_rb/'

    old_policy_total_reward = None
    prev_tstamp_for_db = None
    previous_model_save_path = None
    current_oracle_model = args.initial_model
    rewards = []

    logger.log(f'Waiting for first {iteration_length_s} to make sure enough '
               f'data is gathered for simulation')
    time.sleep(iteration_length_s)
    logger.log('Training loop: entering...')

    while running:
        logger.debug(f'Training loop: iteration {iterations}')

        # start of the current iteration
        iteration_start = time.time()

        # how much time had passed since the start of the training
        iteration_delta = iteration_start - global_start

        current_tstamp_for_db = initial_timestamp + iteration_delta

        training_data_end = current_tstamp_for_db

        if prev_tstamp_for_db:
            training_data_start = prev_tstamp_for_db
        else:
            training_data_start = current_tstamp_for_db - iteration_length_s

        # this needs to happen after training_data_start, when we determine
        # the start of the data we need to have the value from the previous
        # iteration
        prev_tstamp_for_db = current_tstamp_for_db

        updated_extra_args = {
            'iteration_start': iteration_start,
            'training_data_start': training_data_start,
            'training_data_end': training_data_end,
        }
        updated_extra_args.update(extra_args)

        cores_count = get_cores_count_at(training_data_start)

        logger.log(f'Using training data starting at: '
                   f'{training_data_start}-{training_data_end} '
                   f'Initial tstamp: {initial_timestamp}')

        if any([not data_available(v) for k, v in cores_count.items()]):
            initial_vm_cnt = args.initial_vm_count_no_data
            if initial_vm_cnt:
                logger.log(f'Cannot retrieve vm counts from db: {cores_count}. '
                           f'Overwriting with initial_vm_count_no_data: '
                           f'{initial_vm_cnt}'
                           )

                cores_count = {
                    's_cores': initial_vm_cnt,
                    'm_cores': initial_vm_cnt,
                    'l_cores': initial_vm_cnt,
                }

        if all([data_available(v) for k, v in cores_count.items()]):
            logger.log(f'Initial cores: {cores_count}')

            if previous_model_save_path:
                load_path = previous_model_save_path
            else:
                logger.info(f'No model from previous iteration, using the '
                            f'initial one: {args.initial_model}')
                load_path = args.initial_model

            updated_extra_args.update({
                'initial_s_vm_count': math.floor(cores_count['s_cores']/2),
                'initial_m_vm_count': math.floor(cores_count['m_cores']/2),
                'initial_l_vm_count': math.floor(cores_count['l_cores']/2),
                'load_path': load_path,
            })

            env = build_env(args, updated_extra_args)
            # if there is no new env it means there was no training data
            # thus there is no need to train and evaluate
            if env is not None:
                logger.info('New environment built...')
                model = train(args,
                              updated_extra_args,
                              env)

                model_save_path = f'{base_save_path}_{iterations}.zip'
                model_replay_buffer_save_path = f'{base_save_path}_{iterations}_rb/'
                model.save(model_save_path)
                model.save_replay_buffer(model_replay_buffer_save_path)
                previous_model_save_path = model_save_path

                logger.info(f'Test model after {iterations} iterations')
                new_policy_total_reward = test_model(model, env)
                rewards.append(new_policy_total_reward)

                rewards_df = pd.DataFrame(rewards, columns=['reward'])
                rewards_df.to_csv('/best_model/best_model_rewards.csv')

                if old_policy_total_reward is None:
                    best_model = DQN.load(best_model_path)
                    old_policy_total_reward = test_model(best_model, env)

                logger.info(f'Old policy reward: {old_policy_total_reward} '
                            f'new policy reward: {new_policy_total_reward}')
                if new_policy_total_reward > old_policy_total_reward:
                    logger.info('New policy has a higher reward, updating the policy')
                    model.save(best_model_path)
                    model.save_replay_buffer(best_model_replay_buffer_path)
                    old_policy_total_reward = new_policy_total_reward

                # if current_oracle_model:
                #     model.load(current_oracle_model)
                #     old_policy_total_reward = test_model(model, env)
                # else:
                #     old_policy_total_reward = None
                #
                # logger.info(f'Old policy reward: {old_policy_total_reward} '
                #             f'new policy reward: {new_policy_total_reward}')
                # if old_policy_total_reward:
                #     if new_policy_total_reward > old_policy_total_reward:
                #         logger.info('New policy has a higher reward, '
                #                     'updating the policy')
                #         update_oracle_policy(model_save_path)
                #         current_oracle_model = model_save_path
                # else:
                #     logger.info('The old policy do not exist - we choose to '
                #                 'do an update')
                #     update_oracle_policy(model_save_path)
                #     current_oracle_model = model_save_path
            else:
                logger.info('The environment has not changed, training and '
                            'evaluation skipped')
        else:
            logger.log(f'Cannot initialize vm counts - not enough data '
                       f'available. Cores count: {cores_count}')

        iteration_end = time.time()
        iteration_len = iteration_end - iteration_start
        time_fill = iteration_length_s - iteration_len
        if time_fill > 0:
            logger.log(f'Sleeping for {time_fill}')
            time.sleep(time_fill)
        else:
            logger.log(f'Does not make sense to wait... '
                       f'We used {-time_fill}s more than expected')

        iterations += 1

    env.close()
    logger.log('Training loop: ended')


def training_once(args, extra_args):
    logger.log('Training once: starting...')
    env = build_env(args, extra_args)
    model = train(args, extra_args, env)

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
    configure_logger(args.log_path, format_strs=['stdout', 'log', 'csv'])
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.continuous_mode:
        training_loop(args, extra_args)
    else:
        training_once(args, extra_args)


if __name__ == '__main__':
    main()
