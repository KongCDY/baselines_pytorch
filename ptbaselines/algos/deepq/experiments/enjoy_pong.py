import gym
import numpy as np
from ptbaselines.algos import deepq
from ptbaselines.algos.common import torch_utils


def main():
    env = gym.make("PongNoFrameskip-v4")
    env = deepq.wrap_atari_dqn(env)
    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        total_timesteps=0,
        load_path = 'pong_model.pth'
    )

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            action = act.actions(torch_utils.toTensor(obs[None]).float(), stochastic=False)[0].item()
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
