from src.train import train
from src.train import single_rollout
from src.train import APEX_DDPG_DEFAULT_CONFIG
import warnings
from omegaconf import OmegaConf

from toy_models.envs.toy_functions import TF2D_DEFAULT_CONFIG
import toy_models
import click

warnings.filterwarnings("ignore")

# Env Config
env_config = TF2D_DEFAULT_CONFIG
env_config.max_steps=200
env_config.d_factor = 5
env_config.lh_factor = 3
env_config.kernel_bandwidth = 0.1
env_config.density_limit = 0.3
env_config.kernel = 'gaussian'
env_config.density_state = False 
env_config.observables_state  = False 
env_config.parameters_state = True 
env_config.lh_function = 'gaussian'

# Agent config
agent_config = APEX_DDPG_DEFAULT_CONFIG
agent_config.agent.actor_lr = 0.0025
agent_config.agent.critic_lr = 0.0025
agent_config.agent.tau = 0.002
agent_config.memory.batch_size = 64


agent_config.agent.num_train_workers = 3
agent_config.agent.num_eval_workers = 2
agent_config.agent.multi_step_n= 3
agent_config.agent.send_experience_freq =  500
agent_config.agent.q_update_freq = 500

agent_config.noise.name = 'Gaussian'
agent_config.noise.sigma = 0.5
agent_config.noise.decrease = False
agent_config.noise.final_sigma = 0.25
agent_config.noise.greedy_sigma = 0.4
agent_config.epsilon = 0.0
agent_config.agent.prioritized = False
agent_config.agent.learning_starts = 100
agent_config.agent.initial_decreasing_step = 100000
agent_config.agent.final_decreasing_step = 400000

agent_config.agent.split_sigma = True
agent_config.env.name = 'ToyFunction2d-v1'

run_name = 'toy_models/test_density_reward_4'

if __name__ == '__main__':

    train(
            agent_config= agent_config,
            env_config=env_config,
            local_mode=False,
            num_cpus=4,
            run_name = run_name
            )
