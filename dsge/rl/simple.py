import gym
import numpy as np

class ToyEnv(gym.Env):
    def __init__(self, k_init=1.0, A=1.0, delta=0.1, T=2):
        self.A = A
        self.k_init = k_init
        self.delta=delta
        self.T=T
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def step(self, action):
        c=(action[0]+1)
        self.t+=1
        reward = self.utility(c)
        if self.t>=self.T:
            done=True
        else:
            done=False
        info = {}
        self.state = [0]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def reset(self):
        self.state = [0]
        self.t=0
        return np.array(self.state, dtype=np.float32)

    def utility(self, c):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return c

    def output(self, k):
        """
        Output of the firm
        Parameters
        ----------
        k : array_like
            Array of capital stock
        """
        return self.A * k