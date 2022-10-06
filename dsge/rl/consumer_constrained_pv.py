import gym
import numpy as np

from dsge.classical.constrained_pv import ConstrainedPV


class ConstrainedPVRL(ConstrainedPV, gym.Env):
    def __init__(self, W=1.0, R=1.0, beta=0.9, T=10, eps=1e-3):
        super().__init__(W=W, R=R, beta=beta, T=T, eps=eps)
        self.action_space = gym.spaces.Box(low=0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def step(self, action):
        c = (action[0])
        [W, t] = self.state
        c = np.clip(c, 0, W / self.R ** (1 - (t + 1)))
        self.c[t] = c
        W -= self.R ** (1 - (t + 1)) * c
        reward = self.utility(c)
        t += 1
        if t >= self.T:
            done = True
        else:
            done = False
        info = {}
        self.state = [W, t]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def reset(self):
        self.state = [self.W, 0]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = ConstrainedPVRL()
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO

    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta).learn(total_timesteps=100000)
    for k in range(10):
        obs = env.reset()
        dones = False
        while not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
