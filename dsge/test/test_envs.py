import dsge.rl


def test_rl_envs():
    from stable_baselines3.common.env_checker import check_env
    for model in dsge.rl.__all__:  # type: ignore
        print(model)
        env = getattr(dsge.rl, model)()
        check_env(env)
