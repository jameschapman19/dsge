import gym
import numpy as np
from stable_baselines3 import PPO

from dsge.rl.precautionary_savings import PrecautionarySavingsRL


class AmbiguousPrecautionarySavingsRL(PrecautionarySavingsRL):
    def __init__(self, W_0=1.0, beta=0.9, T=10, W_shock=0.5, eps=1e-3):
        super().__init__(W_0=W_0, beta=beta, T=T, W_shock=W_shock, eps=eps)
        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.T_shock = np.random.randint(1, self.T)
        self.t = 0
        self.state = [0, self.W_0]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = AmbiguousPrecautionarySavingsRL()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=100000)
    print(f"total utility: {env.total_utility(env.c)}")
    env.render()
