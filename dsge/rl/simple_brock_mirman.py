import gym
import numpy as np

from dsge.classical.simple_brock_mirman import SimpleBrockMirman


class SimpleBrockMirmanRL(SimpleBrockMirman, gym.Env):

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=10, G=0.02):
        super().__init__(alpha=alpha, beta=beta, K_0=K_0, A_0=A_0, T=T, G=G)
        self.action_space = gym.spaces.Box(low=0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def step(self, action):
        C = (action[0])
        [K, A, t] = self.state
        Y = self.production(A, K)
        C = np.clip(C, 0, Y)
        self.k[t] = K
        self.c[t] = C
        reward = self.utility(C)
        K = Y - C
        t += 1
        if t >= self.T:
            done = True
        else:
            done = False
        info = {}
        self.state = [K, A, t]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def reset(self):
        self.state = [self.K_0, self.A_0, 0]
        return np.array(self.state, dtype=np.float32)


if __name__ == "__main__":
    env = SimpleBrockMirmanRL()
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
