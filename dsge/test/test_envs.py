import pytest
from dsge.rl import BrockMirmanRL,SimpleBrockMirmanRL,ConsumerConstrainedPVRL,ConsumerCapitalAccumulationRL

def test_rl_envs():
    from stable_baselines3.common.env_checker import check_env
    for env in [BrockMirmanRL, SimpleBrockMirmanRL, ConsumerConstrainedPVRL, ConsumerCapitalAccumulationRL]:
        check_env(env())