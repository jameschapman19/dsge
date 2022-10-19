import gym
import numpy as np

from dsge.classical.tfp_shock import TFPShock


class TFPShockRL(TFPShock, gym.Env):
    """Social Planner"""

    def __init__(self, beta=0.96, T=30, alpha=0.35, delta=0.06, rho=0.8, A_0=1.01, A_eps=0, A_bar=1):
        super().__init__(beta=beta, T=T, alpha=alpha, delta=delta, rho=rho, A_0=A_0, A_eps=A_eps, A_bar=A_bar)
        self.action_space = gym.spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.T = T

    def step(self, action):
        c = (action[0])
        [k, a] = self.state
        y = self.production(k, a)
        c *= y
        i = y - c
        utility = self.utility(c)
        self.store(y, c, i, k, a, utility)
        y_, i_, k_, a_ = self.model_step(y, i, k, c, a)
        done = self.step_time()
        info = {}
        self.state = [k_, a_]
        return np.array(self.state, dtype=np.float32), utility.item(), done, info

    def store(self, y, c, i, k, a, u):
        self.y[self.t] = y
        self.c[self.t] = c
        self.i[self.t] = i
        self.k[self.t] = k
        self.a[self.t] = a
        self.u[self.t] = u * self.beta ** self.t

    def reset(self):
        self.t = 0
        self.u = np.zeros(self.T)
        self.state = [self.K_bar, self.A_0]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = TFPShockRL(T=100)
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO

    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=50000)
    obs = env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    env.render()
    import matplotlib.pyplot as plt

    plt.show()
