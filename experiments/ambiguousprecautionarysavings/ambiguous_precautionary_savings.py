from os.path import exists

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO

from dsge.rl import PrecautionarySavingsRL, AmbiguousPrecautionarySavingsRL


def train(env, model_name='precautionary_savings', total_timesteps=100000):
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./log/', gamma=env.beta, seed=42).learn(
        total_timesteps=total_timesteps)
    model.save(model_name)


def run_model(env, model):
    obs = env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    df = env.history
    print(f"RL utility: {env.total_utility(env.c)}")
    return df


def plot_for_different_shocks(df, T):
    fig, axs = plt.subplots(1, T, figsize=(20, 5))
    for t in range(T):
        plot_df = df[df['shock'] == t]
        plot_df = pd.melt(plot_df, id_vars=['time', 'run'], value_vars=['consumption', 'wage', 'savings'])
        gfg = sns.lineplot(data=plot_df, x='time', y='value', hue='variable', ax=axs[t])
        gfg.set_ylim(bottom=0)


def evaluate_rl(runs=10, T=10, model_name='precautionary_savings'):
    model = PPO.load(model_name)
    dfs = []
    for t in range(T):
        for run in range(runs):
            env = PrecautionarySavingsRL(beta=0.99, T_shock=t)
            df = run_model(env, model)
            df['run'] = run
            df['shock'] = t
            dfs.append(df)
    df = pd.concat(dfs)
    plot_for_different_shocks(df, T)


def evaluate_classical(env):
    env.solve()
    print(f"total utility: {env.total_utility(env.c)}")
    env.render()


def main(retrain=False, model_name='precautionary_savings', time_steps=150000):
    env = AmbiguousPrecautionarySavingsRL()
    if retrain:
        train(env, model_name=model_name, total_timesteps=time_steps)
    else:
        if not exists(f"{model_name}.zip"):
            train(env, model_name=model_name)
    evaluate_rl(T=10)
    plt.savefig('precautionary_savings_rl.png')
    plt.show(block=True)


if __name__ == '__main__':
    main()
