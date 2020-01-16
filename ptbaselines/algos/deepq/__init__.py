from ptbaselines.algos.deepq import models  # noqa
from ptbaselines.algos.deepq.deepq import learn
from ptbaselines.algos.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from ptbaselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)