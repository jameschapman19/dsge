from dsge.rl import BrockMirmanRL, SimpleBrockMirmanRL, ConstrainedPVRL, CapitalAccumulationRL


def test_rl_envs():
    from stable_baselines3.common.env_checker import check_env
    for env in [BrockMirmanRL, SimpleBrockMirmanRL, ConstrainedPVRL, CapitalAccumulationRL]:
        check_env(env())
