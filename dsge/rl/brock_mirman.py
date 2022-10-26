import gym
import numpy as np

from dsge.classical.brock_mirman import BrockMirman


class BrockMirmanRL(BrockMirman, gym.Env):
    """Social Planner"""

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=10, G=0.02, b=0.5):
        """

        Parameters
        ----------
        alpha: float
            Capital share of output
        beta: float
            Discount factor
        K_0: float
            Initial capital stock
        A_0: float
            Initial technology level
        T: int
            Number of periods
        G: float
            Growth rate of technology
        """
        super().__init__(alpha=alpha, beta=beta, K_0=K_0, A_0=A_0, T=T, G=G, b=b)
        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def step(self, action):
        spending_rate, l = action[0], action[1]
        [k, a] = self.state
        y = self.production(a, k, N=l)
        c = y * spending_rate
        reward = self.utility(c, l)
        self.store(y, c, l, k)
        k = y * (1 - spending_rate)
        done = self.step_time()
        if not done:
            a = self.A[self.t]
            self.state = [k, a]
        info = {}
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def store(self, y, c, l, k):
        self.y[self.t] = y
        self.c[self.t] = c
        self.l[self.t] = l
        self.k[self.t] = k

    def reset(self):
        self.t = 0
        self.state = [self.K_0, self.A[0]]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = BrockMirmanRL()
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO

    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=100000)
