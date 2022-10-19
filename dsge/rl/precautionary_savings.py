import gym
import numpy as np

from dsge.classical.precautionary_savings import PrecautionarySavings


class PrecautionarySavingsRL(PrecautionarySavings, gym.Env):
    def __init__(self, W=1.0, R=1.0, beta=0.9, T=10, T_shock=5, W_shock=0.5, eps=1e-3):
        super().__init__(W=W, R=R, beta=beta, T=T, T_shock=T_shock, W_shock=W_shock, eps=eps)
        self.action_space = gym.spaces.Box(low=0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def step(self, action):
        c, n = (action[0]), (action[1])
        [s, w, t] = self.state
        c = np.clip(c, 0, s + w * n)
        reward = self.utility(c)
        s = self.model_step(c, s, w)
        self.state = self.store(t, c, s, w)
        t, done = self.step_time(t)
        self.s[t] = s
        w = self.w[t]
        info = {}
        self.state = [s, w, t]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def store(self, t, c, s, w):
        self.s[t] = s
        self.c[t] = c
        self.w[t] = w
        return [s, w, t]

    def reset(self):
        self.state = [0, self.w[0], 0]
        return np.array(self.state, dtype=np.float32)
