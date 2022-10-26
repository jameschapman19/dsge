import gym
import numpy as np
from stable_baselines3 import PPO

from dsge.classical.precautionary_savings import PrecautionarySavings


class AmbiguousPrecautionarySavingsRL(PrecautionarySavings, gym.Env):
    def __init__(self, W=1.0, R=1.0, beta=0.9, T=10, T_shock=5, W_shock=0.5, eps=1e-3):
        super().__init__(W=W, R=R, beta=beta, T=T, T_shock=T_shock, W_shock=W_shock, eps=eps)
        self.action_space = gym.spaces.Box(low=0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def step(self, action):
        c, n = (action[0]), (action[1])
        [s, w] = self.state
        c = np.clip(c, 0, s + w * n)
        reward = self.utility(c)
        s = self.model_step(c, s, w)
        self.state = self.store(c, s, w)
        done = self.step_time()
        if not done:
            w = self.w[self.t]
        info = {}
        self.state = [s, w]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def model_step(self, c, s, w):
        s_ = s + w - c
        return s_

    def model(self, c):
        for t in range(1, self.T):
            self.s[t] = self.model_step(c[t - 1], self.s[t - 1], self.w[t])

    def store(self, c, s, w):
        self.s[self.t] = s
        self.c[self.t] = c
        self.w[self.t] = w
        return [s, w]

    def reset(self):
        self.t = 0
        self.state = [0, self.w[0]]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = AmbiguousPrecautionarySavingsRL()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=100000)
    print(f"total utility: {env.total_utility(env.c)}")
    env.render()
