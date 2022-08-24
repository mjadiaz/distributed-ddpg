from src.train import train
from src.train import single_rollout
from src.train import APEX_DDPG_DEFAULT_CONFIG
import warnings
from omegaconf import OmegaConf

from toy_models.envs.toy_functions import TF2D_DEFAULT_CONFIG
import toy_models
from train import agent_config, run_name, env_config

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    single_rollout(
        agent_config=agent_config,
        env_config=env_config,
        run_name = run_name,
            )
