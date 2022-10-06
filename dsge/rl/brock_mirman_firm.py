import gym
import numpy as np

from dsge.classical.brock_mirman import BrockMirman


class NKRL(BrockMirman, gym.Env):
    """Social Planner"""

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=10, G=0.02, stickiness=0.5):
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
        stickiness: float
            Probability of each firm being able to change prices
        """
        super().__init__(alpha=alpha, beta=beta, K_0=K_0, A_0=A_0, T=T, G=G)
        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def labour_demand(self, C, A, K):
        return ((C / K ** (1 - self.alpha)) ** (1 / (self.alpha))) / A

    def profit(self, Y, W, N, R, K):
        return Y - W * N - (R - 1 + self.delta) * K

    def intermediary_firm_demand(self, P_i, epsilon_p, Y):
        return P_i ** (-epsilon_p) * Y

    def step(self, action):
        """
        Firm takes R and W as given and profit maximises
        """
        [P_i, A, t] = self.state
        spending_rate = action[0]
        Y_i = self.intermediary_firm_demand(P_i, action[0])
        N = self.labour_demand(Y, A, K)
        t += 1
        W = self.alpha * A ** self.alpha * (K / N) ** (1 - self.alpha)
        R = (1 - self.alpha) * (A * N / K) ** self.alpha + (1 - self.delta)
        reward = self.profit(Y, W, N, R, K)
        if t == self.T:
            done = True
        else:
            A = self.A[t]
            done = False
        info = {}
        self.state = [A, t]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def reset(self):
        self.state = [self.A[0], 0]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = BrockMirmanFirmRL()
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
