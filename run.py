import os
import datetime
import gym
import gym_cloudsimplus
import json
import math
import numpy as np
import os
import os.path as osp
import pandas as pd
import re
# import requests
import sys
import time
from collections import defaultdict
from importlib import import_module
import stable_baselines3
import sb3_contrib
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from driver import logger
from driver.common.cmd_util import (
    common_arg_parser,
    parse_unknown_args
)
from driver.common.db import (
    get_cores_count_at,
    get_jobs_between,
    init_dbs,
    get_all_cores_count_at
)
from driver.common.swf_jobs import get_jobs_from_file
from driver.baseline_models import RandomModel, SimpleCPUThresholdModel
from driver.wrappers import TimeLimit


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
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, total_timesteps: int, verbose: int = 1, intermediate_models: int = 5):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.total_timesteps = total_timesteps
        self.intermediate_models = intermediate_models
        self.save_count = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # if self.n_calls % 100 == 0:
        #     self.model.save(self.save_path + '_' + str(self.n_calls))
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)
        freq = self.total_timesteps // self.intermediate_models
        if self.n_calls % freq == 0 or self.n_calls == 1:
            if self.verbose > 0:
                print(f"Saving {self.save_count}th intermediete model to {self.save_path + str(self.save_count)}")
            self.model.save(self.save_path + '_' + str(self.save_count) + '_' + str(self.n_calls))
            self.save_count += 1

        return True


def train(args,
          model,
          extra_args,
          env,
          tensorboard_log,
          use_callback=True
          ):
    total_timesteps = int(args.num_timesteps)
    nsteps = int(args.n_steps)
    logger.info(model)
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=tensorboard_log, total_timesteps=total_timesteps, intermediate_models=4)
    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=callback if use_callback else None)
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
    seed = int(args.seed)
    initial_vm_count = args.initial_vm_count
    simulator_speedup = args.simulator_speedup
    queue_wait_penalty = args.queue_wait_penalty

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
        'observation_history_length': args.observation_history_length,
        'queue_wait_penalty': str(queue_wait_penalty)
    }

    # # Create log dir
    # log_dir = "tmp/"
    # os.makedirs(log_dir, exist_ok=True)

    # # Create and wrap the environment
    # env = Monitor(env, log_dir)
    algo = args.algo
    policy = args.policy
    tensorboard_log = f"/output_models_initial/{algo.lower()}/{policy}/"

    # from gym.wrappers.time_limit import TimeLimit
    logger.info(args)
    def custom_wrapper(env):
        env = TimeLimit(env, 
            max_episode_steps=500, 
            penalty=-0.1
            )
        return env

    env = make_vec_env(env_id=env_id,
                       n_envs=args.num_env or 1,
                       monitor_dir=tensorboard_log,
                       seed=seed,
                       wrapper_class=custom_wrapper,
                       env_kwargs=env_kwargs
                       )
    
    env = VecNormalize(env, norm_obs=False)

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


# def update_oracle_policy(model_save_path):
#     with open(model_save_path, 'rb') as updated_model:
#         updated_model_bytes = updated_model.read()
#         updated_model_encoded = base64.b64encode(updated_model_bytes)
#         updated_model_str = updated_model_encoded.decode('utf-8')

#     req_payload = {
#         "content": updated_model_str,
#         "file_suffix": datetime.datetime.now().isoformat(),
#         "network_type": "lstm",
#     }

#     resp = requests.post(oracle_update_url, json=req_payload)
#     logger.log(f'Model updated ({resp.status_code}): {resp.text}')

def build_dqn_model(AlgoClass, policy, tensorboard_log, env, args):
    return AlgoClass(policy=policy,
                     env=env,
                    #  learning_rate=0.0001,
                     exploration_fraction=0.2,
                    #  batch_size=64,
                    #  buffer_size=10000,
                    #  learning_starts=1000,
                    #  target_update_interval=500,
                     verbose=1,
                     seed=int(args.seed),
                     tensorboard_log=tensorboard_log)

def build_ppo_model(AlgoClass, policy, tensorboard_log, env, n_steps, args):
    return AlgoClass(policy=policy,
                     env=env,
                     n_steps=n_steps,
                     learning_rate=0.00003, # 0.00003
                     vf_coef=1,
                     clip_range_vf=10.0,
                     max_grad_norm=1,
                     gamma=0.95,
                     ent_coef=0.001,
                     clip_range=0.05,
                     verbose=1,
                     seed=int(args.seed),
                     tensorboard_log=tensorboard_log)

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
    n_features = 7
    running = True
    env = None
    algo = args.algo
    policy = args.policy
    n_steps = int(args.n_steps)
    try:
        AlgoClass = getattr(stable_baselines3, algo)
    except:
        AlgoClass = getattr(sb3_contrib, algo)

    algo_tag = f"{algo.lower()}/{policy}"
    base_save_path = os.path.join(base_save_path, algo_tag)
    date_tag = datetime.datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
    base_save_path = os.path.join(base_save_path, date_tag)

    best_model_path = f'/best_model/{algo.lower()}/{policy}/{algo.lower()}_{policy}'
    best_model_replay_buffer_path = f'/best_model/{algo.lower()}/{policy}/best_model_rb/'

    best_policy_total_reward = None
    prev_tstamp_for_db = None
    previous_model_save_path = None
    tensorboard_log = f"/output_models/{algo.lower()}/{policy}/"
    current_oracle_model = args.initial_model
    rewards = []
    mean_episode_lenghts = []

    if 'Cnn' in policy:
        args.observation_history_length = 15
        all_observations = np.zeros((1, 1, 1, args.observation_history_length, n_features))
    else:
        args.observation_history_length = 1
        all_observations = np.zeros((1, args.observation_history_length, n_features))

    logger.log(f'Using {algo} with {policy}')
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

            updated_extra_args.update({
                'initial_s_vm_count': math.floor(cores_count['s_cores'] / 2),
                'initial_m_vm_count': math.floor(cores_count['m_cores'] / 2),
                'initial_l_vm_count': math.floor(cores_count['l_cores'] / 2),
            })

            env = build_env(args, updated_extra_args)
            # if there is no new env it means there was no training data
            # thus there is no need to train and evaluate
            
            if env is not None:
                if previous_model_save_path:
                    if algo == 'DQN':
                        model = AlgoClass.load(path=previous_model_save_path,
                                                env=env,
                                                tensorboard_log=tensorboard_log)
                    else:
                        model = AlgoClass.load(path=previous_model_save_path,
                                                env=env,
                                                tensorboard_log=tensorboard_log)
                else:
                    initial_model_path = f'/initial_model/best_models/{algo.lower()}_{policy}'
                    if args.initial_model == 'use_initial_policy':
                        logger.info(f'No model from previous iteration, using the '
                                    f'initial one')
                        
                        model = AlgoClass.load(path=initial_model_path,
                                               env=env,
                                               tensorboard_log=tensorboard_log)
                    else:
                        logger.info(f'No model from previous iteration and no initial model, creating '
                                    f'new model')
                        if algo == 'DQN':
                            model = build_dqn_model(AlgoClass=AlgoClass, policy=policy, env=env, tensorboard_log=tensorboard_log, args=args)
                        else:
                            model = build_ppo_model(AlgoClass=AlgoClass, policy=policy, env=env, tensorboard_log=tensorboard_log, n_steps=n_steps, args=args)

                logger.info('New environment built...')
                model = train(args,
                              model,
                              updated_extra_args,
                              env,
                              tensorboard_log=tensorboard_log,
                              use_callback=False)

                model_save_path = f'{base_save_path}_{iterations}.zip'
                model.save(model_save_path)
                model.save(best_model_path)
                # model_replay_buffer_save_path = f'{base_save_path}_{iterations}_rb/'
                # model.save_replay_buffer(model_replay_buffer_save_path)
                previous_model_save_path = model_save_path

                logger.info(f'Test model after {iterations} iterations')
                env.reset()
                num_env = args.num_env
                args.num_env = 1
                env = build_env(args, updated_extra_args)
                args.num_env = num_env
                new_policy_total_reward, observations, episode_lenghts, episode_rewards, rewards_per_run, observations_evalute_results, actions = test_model(model, env, n_runs=15)
                rewards.append(new_policy_total_reward)
                mean_episode_lenghts.append(episode_lenghts.mean())

                logger.info(f'Save: training_data, observations and actions.')
                df = pd.DataFrame()
                df['reward'] = rewards
                df['episode_len'] = mean_episode_lenghts
                df.to_csv(f'/best_model/{algo.lower()}/{policy}/training_data.csv')

                # all_observations.append(observations)
                all_observations = np.append(all_observations, observations, axis=0)
                with open(f'/best_model/{algo.lower()}/{policy}/observations.npy', 'wb') as f:
                    np.save(f, all_observations)

                
                with open(f'/best_model/{algo.lower()}/{policy}/actions.npy', 'wb') as f:
                    np.save(f, actions)
       
                if iterations > 0:
                    if best_policy_total_reward is None:
                        best_model = AlgoClass.load(best_model_path)
                        best_policy_total_reward, _, _, _, _, _, _ = test_model(best_model, env)

                    logger.info(f'Best policy reward: {best_policy_total_reward} '
                                f'new policy reward: {new_policy_total_reward}')
                    if new_policy_total_reward >= best_policy_total_reward:
                        logger.info('New policy has a higher reward')
                        # model.save(best_model_path)
                        best_policy_total_reward = new_policy_total_reward
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
    
    algo = args.algo
    policy = args.policy
    tensorboard_log = f"/output_models_initial/{algo.lower()}/{policy}/"
    n_steps = int(args.n_steps)
    n_features = 6
    try:
        AlgoClass = getattr(stable_baselines3, algo)
    except:
        AlgoClass = getattr(sb3_contrib, algo)

    if 'Cnn' in policy:
        args.observation_history_length = 15
        all_observations = np.zeros((1, 1, 1, args.observation_history_length, n_features))
    else:
        args.observation_history_length = 1
        all_observations = np.zeros((1, args.observation_history_length, n_features))

    initial_model_path = f'/initial_model/{algo.lower()}/{policy}/{algo.lower()}_{policy}'

    # build env
    env = build_env(args, extra_args)

    # build model
    if algo == 'DQN':
        model = build_dqn_model(AlgoClass=AlgoClass, policy=policy, env=env, tensorboard_log=tensorboard_log, args=args)
    else:
        model = build_ppo_model(AlgoClass=AlgoClass, policy=policy, env=env, tensorboard_log=tensorboard_log, n_steps=n_steps, args=args)

    import signal
    import sys

    def signal_handler(sig, frame):
        logger.warn(f'Forced stop to the training process!')
        logger.info(f'Test the model')
        env.reset()
        num_env = args.num_env
        args.num_env = 1
        env = build_env(args, extra_args)
        args.num_env = num_env
        new_policy_total_reward, observations, episode_lenghts, rewards, rewards_per_run, observations_evalute_results, actions = test_model(model, env, n_runs=15)

        print(rewards.shape)
        df = pd.DataFrame()
        df['reward'] = rewards_per_run
        df['episode_len'] = episode_lenghts

        logger.info(f'Save: training_data, observations and actions.')
        df.to_csv(f'/initial_model/{algo.lower()}/{policy}/training_data.csv')

        all_observations = np.append(all_observations, observations, axis=0)
        with open(f'/initial_model/{algo.lower()}/{policy}/observations.npy', 'wb') as f:
            np.save(f, all_observations)

        with open(f'/initial_model/{algo.lower()}/{policy}/actions.npy', 'wb') as f:
            np.save(f, actions)
        
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # train the model
    if env:
        logger.info('New environment built...')
        
        logger.info('Started training...')
        model = train(args,
                    model,
                    extra_args,
                    env,
                    tensorboard_log=tensorboard_log)


        model.save(initial_model_path)

        logger.info(f'Test the model')
        env.reset()
        num_env = args.num_env
        args.num_env = 1
        env = build_env(args, extra_args)
        args.num_env = num_env
        new_policy_total_reward, observations, episode_lenghts, rewards, rewards_per_run, observations_evalute_results, actions = test_model(model, env, n_runs=15)

        print(rewards.shape)
        df = pd.DataFrame()
        df['reward'] = rewards_per_run
        df['episode_len'] = episode_lenghts
        df.to_csv(f'/initial_model/{algo.lower()}/{policy}/training_data.csv')

        all_observations = np.append(all_observations, observations, axis=0)
        with open(f'/initial_model/{algo.lower()}/{policy}/observations.npy', 'wb') as f:
            np.save(f, all_observations)

        with open(f'/initial_model/{algo.lower()}/{policy}/actions.npy', 'wb') as f:
            np.save(f, actions)

    env.close()
    logger.log('Training once: ended')


def evaluate_continuous(args, extra_args):
    logger.log('Initializing databases')
    init_dbs()
    logger.log('Initialized databases')
    iteration_length_s = args.iteration_length_s
    # global training start
    global_start = time.time()
    initial_timestamp = global_start if args.initial_timestamp < 0 else args.initial_timestamp
    prev_tstamp_for_db = None

    logger.log(f'Waiting for first {iteration_length_s} to make sure enough '
               f'data is gathered for simulation')
    time.sleep(iteration_length_s)
    logger.log('Evaluation loop: entering...')

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

        updated_extra_args.update({
            'initial_s_vm_count': math.floor(cores_count['s_cores'] / 2),
            'initial_m_vm_count': math.floor(cores_count['m_cores'] / 2),
            'initial_l_vm_count': math.floor(cores_count['l_cores'] / 2),
        })
        env = build_env(args, updated_extra_args)

        models = {
            # 'RandomModel': [RandomModel(action_space=env.action_space)],
            # 'SimpleCPUThresholdModel': [SimpleCPUThresholdModel(action_space=env.action_space)],
            # 'DQN': {'MlpPolicy', 'CnnPolicy'},
            # 'PPO': {'MlpPolicy', 'CnnPolicy'},
            'RecurrentPPO': {'MlpLstmPolicy', 
            # 'CnnLstmPolicy'
            },
        }
        baseline_models_list = ['RandomModel']
        evaluation_results = pd.DataFrame([])
        for algo, policies in models.items():
            if algo not in ['RandomModel', 'SimpleCPUThresholdModel']:
                try:
                    AlgoClass = getattr(stable_baselines3, algo)
                except:
                    AlgoClass = getattr(sb3_contrib, algo)
            for policy in policies:
                if algo not in ['RandomModel', 'SimpleCPUThresholdModel']:
                    # X = np.load(f'/best_model/{algo.lower()}/{policy}/observations.npy')
                    X = np.load(f'/initial_model/best_models/{algo.lower()}/{policy}/observations.npy')
                    args.observation_history_length = f'{X.shape[-2]}'
                else:
                    args.observation_history_length = '1'
                env = build_env(args, updated_extra_args)
                if env is not None:
                    logger.log(f'Loading best {algo} model, with {policy}')
                    if algo not in ['RandomModel', 'SimpleCPUThresholdModel']:
                        # model = AlgoClass.load(path=f'/best_model/{algo.lower()}/{policy}/best_model.zip',
                        #                     env=env)

                        # model = AlgoClass.load(path=f'/best_model/best_models/{algo.lower()}_{policy}.zip',
                        #                     env=env)
                        model = AlgoClass.load(path=f'/initial_model/best_models/{algo.lower()}_{policy}', env=env)
                    else:
                        model = policy
                    logger.log('Evaluating...')
                    logger.log(env.reset())
                    mean_reward, observations, episode_lenghts, rewards, rewards_per_run, observations_evalute_results, actions   = test_model(model, env, n_runs=10)
                    
                    if algo not in ['RandomModel', 'SimpleCPUThresholdModel']:
                        evaluation_results[f'rewards_per_run_{algo}_{policy}'] = rewards_per_run
                        evaluation_results[f'episode_lenghts_{algo}_{policy}'] = episode_lenghts
                    else:
                        evaluation_results[f'rewards_per_run_{policy}'] = rewards_per_run
                        evaluation_results[f'episode_lenghts_{policy}'] = episode_lenghts

                    with open(f'/best_model/observations/observations_{args.observation_history_length}.npy', 'wb') as f:
                        np.save(f, observations_evalute_results)

                    with open(f'/best_model/rewards/rewards_{algo.lower()}_{policy}.npy', 'wb') as f:
                        np.save(f, observations_evalute_results)

        evaluation_results.to_csv(f'/best_model/evaluation_results.csv')

    env.close()
    logger.log('Evaluation ended')

def evaluate_sample(args, extra_args):
    # build env
    env = build_env(args, extra_args)

    if env:
        models = {
            # 'RandomModel': [RandomModel(action_space=env.action_space)],
            # 'SimpleCPUThresholdModel': [SimpleCPUThresholdModel(action_space=env.action_space)],
            # 'DQN': {
            #     'MlpPolicy', 
            #     'CnnPolicy'
            #     },
            # 'PPO': {
            #     'MlpPolicy', 
            #     'CnnPolicy'
            #     },
            'RecurrentPPO': {
                'MlpLstmPolicy', 
                # 'CnnLstmPolicy'
                },
        }
        evaluation_results = pd.DataFrame([])
        for algo, policies in models.items():
            if algo not in ['RandomModel', 'SimpleCPUThresholdModel']:
                try:
                    AlgoClass = getattr(stable_baselines3, algo)
                except:
                    AlgoClass = getattr(sb3_contrib, algo)
            for policy in policies:
                if algo not in ['RandomModel', 'SimpleCPUThresholdModel']:
                    X = np.load(f'/initial_model/best_models/{algo.lower()}/{policy}/observations.npy')
                    args.observation_history_length = f'{X.shape[-2]}'
                else:
                    args.observation_history_length = '1'
                env = build_env(args, extra_args)
                if env is not None:
                    logger.log(f'Loading best {algo} model, with {policy} and observation history length {args.observation_history_length}')
                    if algo not in ['RandomModel', 'SimpleCPUThresholdModel']:
                        model = AlgoClass.load(path=f'/initial_model/best_models/{algo.lower()}_{policy}',
                                            env=env)

                        # model = AlgoClass.load(path=f'/initial_model/best_models/{algo.lower()}_{policy}.zip',
                        #                     env=env)
                    else:
                        model = policy
                    logger.log('Evaluating...')
                    logger.log(env.reset())
                    mean_reward, observations, episode_lenghts, rewards, rewards_per_run, observations_evalute_results, actions  = test_model(model, env, n_runs=10)
                    
                    if algo not in ['RandomModel', 'SimpleCPUThresholdModel']:
                        evaluation_results[f'rewards_per_run_{algo}_{policy}'] = rewards_per_run
                        evaluation_results[f'episode_lenghts_{algo}_{policy}'] = episode_lenghts
                    else:
                        evaluation_results[f'rewards_per_run_{policy}'] = rewards_per_run
                        evaluation_results[f'episode_lenghts_{policy}'] = episode_lenghts

                    with open(f'/initial_model/observations/observations_{algo.lower()}_{policy}_{args.observation_history_length}.npy', 'wb') as f:
                        np.save(f, np.array(observations))

                    with open(f'/initial_model/rewards/rewards_{algo.lower()}_{policy}.npy', 'wb') as f:
                        np.save(f, rewards)

                    with open(f'/initial_model/actions/actions_{algo.lower()}_{policy}.npy', 'wb') as f:
                        np.save(f, actions)

        evaluation_results.to_csv(f'/initial_model/evaluation_results.csv')

    env.close()
    logger.log('Evaluation ended')

def save_workload(args, extra_args):
    pass

def main():
    args = sys.argv
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    configure_logger(args.log_path, format_strs=['stdout', 'log', 'csv'])
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.evaluate_continuous_mode:
        evaluate_continuous(args, extra_args)
        return
    elif args.evaluate_mode:
        evaluate_sample(args, extra_args)
        return
    elif args.continuous_mode:
        training_loop(args, extra_args)
        return
    else:
        training_once(args, extra_args)
    # if args.save_workload_mode:
    #     save_workload(args, extra_args)
        return

    sys.exit(0)
if __name__ == '__main__':
    main()
