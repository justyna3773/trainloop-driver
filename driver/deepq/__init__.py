from driver.deepq import models  # noqa
from driver.deepq.build_graph import build_act, build_train  # noqa
from driver.deepq.deepq import learn, load_act  # noqa
from driver.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from driver.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
