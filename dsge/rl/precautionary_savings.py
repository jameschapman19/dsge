import gym
import numpy as np

from dsge.classical.precautionary_savings import PrecautionarySavings


class PrecautionarySavingsRL(PrecautionarySavings, gym.Env):
    def __init__(self, W_0=1.0, beta=0.9, T=10, T_shock=5, W_shock=0.5, eps=1e-3):
        super().__init__(W_0=W_0, beta=beta, T=T, T_shock=T_shock, W_shock=W_shock, eps=eps)
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

    def store(self, c, s, w):
        self.s[self.t] = s
        self.c[self.t] = c
        self.w[self.t] = w
        return [s, w]

    def reset(self):
        self.t = 0
        self.state = [0, self.W_0]
        return np.array(self.state, dtype=np.float32)
