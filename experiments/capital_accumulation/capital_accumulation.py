from os.path import exists

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO

from dsge.classical.capital_accumulation import CapitalAccumulation
from dsge.rl import CapitalAccumulationRL


def uncertainty_plot(df):
    plot_df = pd.melt(df, id_vars=['time', 'run'], value_vars=['capital', 'consumption'])
    plt.figure()
    gfg = sns.lineplot(data=plot_df, x='time', y='value', hue='variable')
    gfg.set_ylim(bottom=0)


def train(env, model_name='capital_accumulation_demo', total_timesteps=1):
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
    print(f"total utility: {env.total_utility(env.c)}")
    return df


def evaluate_rl(env, runs=10, model_name='capital_accumulation_demo'):
    model = PPO.load(model_name, env=env)
    dfs = []
    for i in range(runs):
        df = run_model(env, model)
        df['run'] = i
        dfs.append(df)
    df = pd.concat(dfs)
    uncertainty_plot(df)


def evaluate_classical(env):
    env.solve()
    print(f"total utility: {env.total_utility(env.c)}")
    env.render()


def main(retrain=False, model_name='capital_accumulation_demo', train_timesteps=100000):
    env = CapitalAccumulationRL()
    if retrain:
        train(env, model_name=model_name, total_timesteps=train_timesteps)
    else:
        if not exists(f"{model_name}.zip"):
            train(env, model_name=model_name, total_timesteps=train_timesteps)
    evaluate_rl(env, runs=10)
    plt.savefig('capital_accumulation.png')
    classical = CapitalAccumulation()
    evaluate_classical(classical)
    plt.savefig('capital_accumulation_classical.png')


if __name__ == '__main__':
    main()
