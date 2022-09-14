import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from dsge.classical.consumer_capital_accumulation import ConsumerCapitalAccumulation

class ConsumerCapitalAccumulationRL(ConsumerCapitalAccumulation,gym.Env):
    def __init__(self, A=1.0, T=10, delta=0.1, k_init=1.0):
        super().__init__(A=A, T=T, delta=delta, k_init=k_init)
        self.action_space = gym.spaces.Box(low=0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32) 

    def step(self, action):
        c=(action[0])
        [k,t]=self.state
        c = np.clip(c,0, self.output(k)+(1-self.delta)*k)
        k = self.output(k) - c + (1 - self.delta) * k
        t+=1
        reward = self.utility(c)
        if t>=self.T:
            done=True
        else:
            done=False
        info = {}
        self.state = [k.item(), t]
        return np.array(self.state, dtype=np.float32), reward.item(), done, info

    def reset(self):
        self.state = [self.k_init, 0]
        return np.array(self.state, dtype=np.float32)
        


if __name__ == "__main__":
    env = ConsumerCapitalAccumulationRL()
    from stable_baselines3.common.env_checker import check_env

    check_env(env)
    from stable_baselines3 import PPO, SAC, DDPG

    # Define and Train the agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/').learn(total_timesteps=20000)
    for k in range(10):
        obs = env.reset()
        dones=False
        while not dones:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
