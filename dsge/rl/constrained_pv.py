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
        [w, t] = self.state
        c = np.clip(c, 0, w / self.R ** (1 - (t + 1)))
        reward = self.utility(c)
        w_ = self.model_step(t, w, c)
        self.store(t, c, w)
        t, done = self.step_time(t)
        info = {}
        self.state = [w_, t]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def store(self, t, c, w):
        self.c[t] = c
        self.w[t] = w

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
