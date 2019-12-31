import gym
import numpy as np

from ptbaselines.algos import deepq
from ptbaselines.algos.common import torch_utils


def main():
    env = gym.make("CartPole-v0")
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="cartpole_model.pth")

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
