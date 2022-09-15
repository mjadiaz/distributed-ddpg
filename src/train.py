from omegaconf import OmegaConf, DictConfig
import numpy as np
import os
import torch
import gym
import ray

from src.actor import Worker
from src.memory import PrioritizedReplayBuffer
from src.memory import ExperienceReplayMemory
from src.learner import Learner
from src.parameter_server import ParameterServer
from src.parameter_server import NewParameterServer
from src.utils import Writer
from src.networks import Actor


APEX_DDPG_DEFAULT_CONFIG = {
        'agent': {
            'actor_lr': 0.001,
            'critic_lr': 0.001,
            'tau': 0.002,
            'gamma': 0.99,
            'save_path': 'results/ddpg',
            'prioritized': False,
            'num_train_workers': 3,
            'num_eval_workers': 2,
            'max_exploration_eps': 1.0,
            'split_sigma': True,
            'learning_starts': 100,
            'max_samples': 5e6,
            'timesteps_per_iteration': 1000,
            'multi_step_n': 3,
            'q_update_freq':100,
            'send_experience_freq':100,
            },
        'env': {
            'name': 'Pendulum-v1',
            },
        'memory': {
            'batch_size': 64,
            'max_size': 1e6,
            'rank_based': False,
            'alpha': 0.7,
            'beta':0.5,
            'beta_rate': 0.9995
            },
        'noise': {
            'name': 'Gaussian',
            'theta': 0.2,
            'sigma': 0.15, #OU 
            'dt': 1e-3,
            'scale': 1.0,
            'epsilon': 0.3,
            'greedy_sigma': 0.0,
            'decrease': False,
            'final_sigma': 0.02,
            'initial_decreasing_step': 8000,
            'final_decreasing_step': 12000,
            },
        'running_on': 'M1'
        }

APEX_DDPG_DEFAULT_CONFIG = OmegaConf.create(APEX_DDPG_DEFAULT_CONFIG)

def add_env_data(config, env_name,env_config=None):
    if env_config is not None:
        env = gym.make(env_name, env_config=env_config)
    else:
        env = gym.make(env_name)
    config.env.state_dimension = env.observation_space.shape[0]
    config.env.action_dimension = env.action_space.shape[0]
    config.env.action_min = env.action_space.low.tolist()
    config.env.action_max = env.action_space.high.tolist()

def train(
        agent_config=APEX_DDPG_DEFAULT_CONFIG,
        env_config=None,
        local_mode=False,
        num_cpus=1,
        run_name = 'run_0',
        checkpoint_name = None
        ):
    agent_config = agent_config
    agent_config.agent.save_path = 'runs/'+run_name
    num_train_workers= agent_config.agent.num_train_workers
    num_eval_workers = agent_config.agent.num_eval_workers
    max_samples = agent_config.agent.max_samples
    timesteps_per_iteration = agent_config.agent.timesteps_per_iteration
    add_env_data(
            agent_config,
            agent_config.env.name,
            env_config=env_config
            )
    # Ray init
    ray.init(local_mode=local_mode, num_cpus=num_cpus)

    # Initialize ray-actors
    writer = Writer("runs/"+run_name)
    parameter_server = NewParameterServer.remote(agent_config)
    if agent_config.agent.prioritized:
        replay_buffer = PrioritizedReplayBuffer.remote(agent_config.memory)
    else:
        replay_buffer = ExperienceReplayMemory.remote(agent_config.memory)

    learner = Learner.remote(
            agent_config,
            replay_buffer,
            parameter_server
            )
    training_actors_ids = []
    eval_actors_ids = []
    


    for i in range(num_train_workers):
        # Distribute epsilon to have a diverse exploration
        if agent_config.agent.split_sigma:
            eps = agent_config.noise.sigma\
                    * i/num_train_workers
        else:
            eps = agent_config.noise.sigma
        # Initialize the actors
        actor = Worker.remote(
                'train-actor:'+str(i),
                replay_buffer,
                parameter_server,
                agent_config,
                eps,
                env_config=env_config
                )
        # Start collecting experiences
        actor.sample.remote()
        # Save the actor ids
        training_actors_ids.append(actor)

    for i in range(num_eval_workers):
        eps = 0
        actor = Worker.remote(
                'eval-actor:'+str(i),
                replay_buffer,
                parameter_server,
                agent_config,
                eps,
                eval = True,
                env_config=env_config,
                )
        # Start evaluating 
        #actor.sample.remote()
        eval_actors_ids.append(actor)

    # Start the learning process
    learner.start_learning.remote()

    total_samples = 0
    best_eval_mean_return = - np.infty
    eval_mean_rewards = []

    while total_samples < max_samples:
        tsid = replay_buffer.get_total_samples.remote()
        new_total_samples = ray.get(tsid)
        if (new_total_samples - total_samples
                >= timesteps_per_iteration):
            # If the new samples are more than
            # the iteration steps
            total_samples = new_total_samples
            print(f'training: total samples {total_samples}')

            parameter_server.set_eval_weights.remote()
            eval_sampling_ids = []
            for eval_actor in eval_actors_ids:
                sid = eval_actor.sample.remote()
                eval_sampling_ids.append(sid)
            eval_results = ray.get(eval_sampling_ids)
            eval_results = np.array(eval_results)
            eval_rewards = eval_results[:,0]
            eval_lengths = eval_results[:,1]
            print(eval_results)
            eval_mean_return = np.mean(eval_rewards)
            eval_mean_length= np.mean(eval_lengths)
            print(f'eval returns: {eval_rewards}')
            print(f'eval mean (over workers) return per episode: {eval_mean_return}')
            print(f'eval mean (over workers) length per episode: {eval_mean_length}')
            writer.add_scalar(
                    "mean_return/total_steps",
                    eval_mean_return,
                    total_samples
                    )
            writer.add_scalar(
                    'mean_ep_len/total_steps',
                    eval_mean_length,
                    total_samples
                    )
            if eval_mean_return > best_eval_mean_return:
                print("Model has improved. Saving the model")
                best_eval_mean_return = eval_mean_return
                checkpoint_name = 'checkpoint' if checkpoint_name is None else checkpoint_name
                parameter_server.save_eval_weights.remote(path=checkpoint_name)

    print('Finishing the training.')
    for actor in training_actors_ids:
        actor.stop.remote()
    learner.stop.remote()

def single_rollout(
        agent_config=APEX_DDPG_DEFAULT_CONFIG,
        env_config=None,
        run_name = 'run_0',
        n_episodes = 1,
        render=True,
        checkpoint_name=None
        ):
    def select_action(actor, state, hp):
        state = torch.tensor(state).float().to(actor.device)
        action = actor(state)
        action = np.clip(
            action.cpu().detach().numpy(),
            hp.env.action_min,
            hp.env.action_max
            )
        return action
    agent_config = agent_config
    agent_config.agent.save_path = 'runs/'+run_name
    add_env_data(
            agent_config,
            agent_config.env.name,
            env_config=env_config
            )

    actor = Actor(agent_config)
    checkpoint_name = 'checkpoint' if checkpoint_name is None else checkpoint_name
    complete_path = os.path.join(agent_config.agent.save_path, checkpoint_name)
    actor.load_model(agent_config.agent.save_path, checkpoint_name)
    if env_config:
        env = gym.make(agent_config.env.name, env_config=env_config)
    else:
        env = gym.make(agent_config.env.name)
    
    for i in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            action = select_action(actor, state, agent_config)
            new_state, reward, done, info = env.step(action)
            state = new_state
            episode_reward += reward
            episode_length += 1
            if render:
                env.render()
        print("Episode Return:", episode_reward)
        print("Episode Length:", episode_length)

