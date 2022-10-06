import gym
import numpy as np

from dsge.classical.brock_mirman import BrockMirman


class BrockMirmanRL(BrockMirman, gym.Env):
    """Social Planner"""

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=10, G=0.02):
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
        super().__init__(alpha=alpha, beta=beta, K_0=K_0, A_0=A_0, T=T, G=G)
        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.l = np.zeros(self.T)

    def step(self, action):
        spending_rate = action[0]
        l = action[1]
        [K, A, t] = self.state
        Y = self.production(A, K, l)
        C = Y * spending_rate
        reward = self.utility(C, l)
        self.l[t] = l
        self.k[t] = K
        self.c[t] = C
        K = self.capital_accumulation(K, Y, C)
        t += 1
        if t == self.T:
            done = True
        else:
            A = self.A[t]
            done = False
        info = {}
        self.state = [K, A, t]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def reset(self):
        self.state = [self.K_0, self.A[0], 0]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = BrockMirmanRL()
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO

    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=200000)
    for k in range(10):
        obs = env.reset()
        dones = False
        while not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
