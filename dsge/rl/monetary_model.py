import gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from dsge._base import _BaseDSGE


class MonetaryModelRL(_BaseDSGE, gym.Env):
    """Social Planner"""

    def __init__(self, beta=0.99, sigma=3, eta=0.001, phi=1, epsilon_tau_dev=0.0005, epsilon_R_dev=0.0005,
                 epsilon_y_dev=0.0005, A=1.3, pi_star=1.01, chi=0.1, gamma=0, T=10):
        super().__init__(beta, T)
        self.action_space = gym.spaces.Box(low=1e-1, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
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
        self.b_act = np.zeros(T)
        self.c_act = np.zeros(T)
        self.n = np.zeros(T)
        self.y = np.zeros(T)
        self.pi = np.zeros(T)
        self.c = np.zeros(T)
        self.b = np.zeros(T)
        self.R = np.zeros(T)
        self.tau = np.zeros(T)
        self.m = np.zeros(T)
        self.u = np.zeros(T)
        self.w = np.zeros(T)

    def ir_rule(self, pi, R_star=2):
        return (R_star - 1) * (pi / self.pi_star) ** (self.A * R_star / (R_star - 1))

    def shock(self):
        return [self.rng.normal(1, self.epsilon_tau_dev), self.rng.lognormal(0, self.epsilon_R_dev), self.rng.normal(1,
                                                                                                                     self.epsilon_y_dev)]

    def utility(self, c, m, n):
        return c ** (1 - self.sigma) / (1 - self.sigma) + self.chi * m ** (1 - self.sigma) / (1 - self.sigma) - n ** (
                1 + self.phi) / (1 + self.phi)

    def step(self, action):
        [m, b, pi, c, n, epsilon_tau, epsilon_R, epsilon_y] = self.state
        c_act, b_act, n_ = (action[0]), (action[1]), (action[2])
        y_ = epsilon_y * n ** (1 - self.eta)
        w_ = (1 - self.eta) * epsilon_y * n ** (-self.eta)
        pi_ = c_act / y_
        c_ = c_act / pi
        b_ = b_act / pi
        R_ = 1 + epsilon_R * self.ir_rule(pi_)
        tau_ = self.gamma * b + epsilon_tau
        m_ = m / pi_ + R_ * b / pi_ - b_ - tau_
        reward = self.utility(c_, m_, n)
        self.store(b_act, c_act, n, y_, pi_, c_, b_,
                   R_, tau_, m_, w_, reward)
        epsilon_tau, epsilon_R, epsilon_y = self.shock()
        done = self.step_time()
        info = {}
        self.state = [m_, b_, pi_, c_, n_, epsilon_tau, epsilon_R, epsilon_y]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def store(self, b_act, c_act, n, y, pi, c, b, R, tau, m, w, u):
        self.b_act[self.t] = b_act
        self.c_act[self.t] = c_act
        self.n[self.t] = n
        self.y[self.t] = y
        self.pi[self.t] = pi
        self.c[self.t] = c
        self.b[self.t] = b
        self.R[self.t] = R
        self.tau[self.t] = tau
        self.m[self.t] = m
        self.u[self.t] = u
        self.w[self.t] = w

    def reset(self):
        self.t = 0
        self.state = [self.rng.lognormal(), self.rng.lognormal(), self.rng.lognormal(), self.rng.lognormal(),
                      self.rng.lognormal()] + self.shock()
        return np.array(self.state, dtype=np.float32)

    @property
    def labor_supply(self):
        return self.w[-1] - self.c[-1] ** self.sigma * self.n[-1] ** self.phi

    @property
    def money_demand(self):
        return self.m[-1] - self.chi ** (1 / self.sigma) * self.c[-1] * ((self.R[-1] - 1) / self.R[-1]) ** (
                -1 / self.sigma)


class FOCCallback(EvalCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(FOCCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('FOC: labor supply', self.locals['env'].envs[0].labor_supply)
        self.logger.record('FOC: money demand', self.locals['env'].envs[0].money_demand)
        self.logger.dump(1)
        return True


if __name__ == "__main__":
    env = MonetaryModelRL(T=100)
    eval_env = MonetaryModelRL(T=100)
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=100, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=100, callback_after_eval=stop_train_callback, verbose=1)
    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=100000,
                                                                                             callback=[eval_callback])
