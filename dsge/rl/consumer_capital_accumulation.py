import gym
import matplotlib.pyplot as plt
import numpy as np

from dsge.classical.capital_accumulation import CapitalAccumulation


class CapitalAccumulationRL(CapitalAccumulation, gym.Env):
    def __init__(self, A=1.0, T=10, delta=0.1, K_0=1.0, beta=0.5):
        super().__init__(A=A, T=T, delta=delta, K_0=K_0, beta=beta)
        self.action_space = gym.spaces.Box(low=0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def step(self, action):
        c = (action[0])
        [k, t] = self.state
        c = np.clip(c, 0, self.output(k) + (1 - self.delta) * k)
        self.k[t] = k
        self.c[t] = c
        k = self.output(k) - c + (1 - self.delta) * k
        t += 1
        reward = self.utility(c)
        if t >= self.T:
            done = True
        else:
            done = False
        info = {}
        self.state = [k.item(), t]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def reset(self):
        self.k = np.zeros(self.T)
        self.c = np.zeros(self.T)
        self.state = [self.K_0, 0]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = CapitalAccumulationRL()
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO

    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=50000)
    dfs = []
    for k in range(10):
        obs = env.reset()
        dones = False
        while not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
        df = env._history()
        print(f"total utility: {env.total_utility(env.c)}")
        env.render()
        plt.show()
