from driver.common import set_global_seeds
from driver.common.runners import AbstractEnvRunner
from driver import logger


def learn(env, seed, total_timesteps, nsteps=2048, **kwargs):
    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    class SimpleRunner(AbstractEnvRunner):

        def __init__(self, *, env, model, nsteps):
            super().__init__(env=env, model=model, nsteps=nsteps)

        def run(self):
            for _ in range(self.nsteps):
                action = self.model.step()
                _, _, done, _ = self.env.step(action)

                if done:
                    logger.info("The episode is done...")

    logger.info(f"Starting random algorithm (nsteps: {nsteps}")
    logger.info(f"Observation space: {ob_space}")
    logger.info(f"Action space: {ac_space}")

    class RandomModel:

        def __init__(self, ac_space):
            self.initial_state = {}
            self.ac_space = ac_space

        def step(self):
            return self.ac_space.sample()

    model = RandomModel(ac_space=ac_space)

    # Instantiate the runner object
    runner = SimpleRunner(env=env, model=model, nsteps=nsteps)

    epochs = total_timesteps // nenvs // nsteps

    for i in range(epochs):
        logger.info(f"Epoch {i}")
        runner.run() #pylint: disable=E0632
