import gym
import numpy as np

from dsge.classical.capital_accumulation import CapitalAccumulation


class CapitalAccumulationRL(CapitalAccumulation, gym.Env):
    def __init__(self, A=1.0, T=10, delta=0.1, K_0=1.0, beta=0.5):
        super().__init__(A=A, T=T, delta=delta, K_0=K_0, beta=beta)
        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def step(self, action):
        c = (action[0])
        [k] = self.state
        c *= self.output(k) + (1 - self.delta) * k
        reward = self.utility(c)
        self.store(k, c)
        k = self.model_step(k, c)
        done = self.step_time()
        info = {}
        self.state = [k]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def store(self, k, c):
        self.k[self.t] = k
        self.c[self.t] = c

    def reset(self):
        self.t = 0
        self.state = [self.K_0]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = CapitalAccumulationRL()
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO

    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=2)
