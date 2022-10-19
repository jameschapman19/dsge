import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MonetaryModelRL(gym.Env):
    """Social Planner"""

    def __init__(self, beta=0.99, sigma=3, eta=0.001, phi=1, epsilon_tau_dev=0.0005, epsilon_R_dev=0.0005,
                 epsilon_y_dev=0.0005, A=1.3, pi_star=1.01, chi=0.1, gamma=0, T=10):
        self.action_space = gym.spaces.Box(low=1e-1, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.epsilon_tau_dev = epsilon_tau_dev
        self.epsilon_R_dev = epsilon_R_dev
        self.epsilon_y_dev = epsilon_y_dev
        self.A = A
        self.pi_star = pi_star
        self.beta = beta
        self.sigma = sigma
        self.eta = eta
        self.phi = phi
        self.chi = chi
        self.gamma = gamma
        self.T = T

    def ir_rule(self, pi, R_star=2):
        return (R_star - 1) * (pi / self.pi_star) ** (self.A * R_star / (R_star - 1))

    def shock(self):
        return [self.rng.normal(1, self.epsilon_tau_dev), self.rng.lognormal(0, self.epsilon_R_dev), self.rng.normal(1,
                                                                                                                     self.epsilon_y_dev)]

    def utility(self, c, m, n):
        return c ** (1 - self.sigma) / (1 - self.sigma) + self.chi * m ** (1 - self.sigma) / (1 - self.sigma) - n ** (
                1 + self.phi) / (1 + self.phi)

    def step(self, action):
        c_act, b_act, n_ = action[0], action[1], action[2]
        [m, b, pi, _, _, _, _, epsilon_tau, epsilon_R, epsilon_y] = self.state
        y_ = epsilon_y * n_ ** (1 - self.eta)
        w_ = (1 - self.eta) * epsilon_y * n_ ** (-self.eta)
        pi_ = c_act / y_
        c_ = c_act / pi
        b_ = b_act / pi
        R_ = 1 + epsilon_R * self.ir_rule(pi_)
        tau_ = self.gamma * b + epsilon_tau
        m_ = m / pi_ + R_ * b / pi_ - b_ - tau_
        reward = self.utility(c_, m_, n_)
        epsilon_tau, epsilon_R, epsilon_y = self.shock()
        self.t += 1
        if self.t == self.T:
            done = True
        else:
            done = False
        info = {}
        self.state = [m_, b_, pi_, c_, n_, w_, R_, epsilon_tau, epsilon_R, epsilon_y]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def reset(self):
        self.t = 0
        self.state = [self.rng.lognormal(), self.rng.lognormal(), self.rng.lognormal(), self.rng.lognormal(),
                      self.rng.lognormal(), self.rng.lognormal(), self.rng.lognormal()] + self.shock()
        return np.array(self.state, dtype=np.float32)


class FOCCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, sigma, phi, chi, beta, verbose=0):
        self.sigma = sigma
        self.phi = phi
        self.chi = chi
        self.beta = beta
        super(FOCCallback, self).__init__(verbose)

    def labor_supply(self, w, c, n):
        return w - c ** self.sigma * n ** self.phi

    def money_demand(self, m, c, R):
        return m - self.chi * c * ((R - 1) / R) ** (-1 / self.sigma)

    def euler_equation(self, c, c_, R, pi_):
        return 1 - self.beta * (c_ / c) ** (-self.sigma) * R / pi_

    def _on_step(self) -> bool:
        [m, _, _, c, n, w, R, _, _, _] = self.locals['new_obs'].T
        self.logger.record('FOC: labor supply', self.labor_supply(w, c, n))
        self.logger.record('FOC: money demand', self.money_demand(m, c, R))
        self.logger.dump(1)
        return True


if __name__ == "__main__":
    env = MonetaryModelRL()
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO

    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=200000,
                                                                                             callback=FOCCallback(
                                                                                                 env.sigma, env.phi,
                                                                                                 env.chi, env.beta))
    for k in range(10):
        obs = env.reset()
        dones = False
        while not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
