import gym

from ptbaselines.algos import deepq
from ptbaselines.algos.common import models


def main():
    env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    act = deepq.learn(
        env,
        network=models.mlp(input_size = env.observation_space.shape, num_hidden=64, num_layers=1),
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=10,
        param_noise=False
    )
    print("Saving model to mountaincar_model.pth")
    act.save("mountaincar_model.pth")


if __name__ == '__main__':
    main()
