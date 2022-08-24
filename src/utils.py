import torch
import os 
import numpy as np
from src.memory import experience
import gym
import toy_models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import ray

class Writer:
    def __init__(self, run_name):
        self.writer = SummaryWriter(run_name)
    def add_scalar(self, name, value, n_iter):
        self.writer.add_scalar(name, value, n_iter)

def batch_to_tensor(experiences_batch, batch_size, device, prioritized=False):
    samples = experiences_batch

    obs = np.array(
            [sample[0] for sample in samples]
            ).reshape(batch_size,-1)
    actions = np.array(
            [sample[1] for sample in samples]
            ).reshape(batch_size, -1)
    rewards = np.array(
            [sample[2] for sample in samples]
            ).reshape(batch_size,)
    last_obs = np.array(
            [sample[3] for sample in samples]
            ).reshape(batch_size,-1)
    dones = np.array(
            [sample[4] for sample in samples]
            ).reshape(batch_size,)
    gammas = np.array(
            [sample[5] for sample in samples]
            ).reshape(batch_size,)

    obs = torch.tensor(obs).float().to(device) 
    actions = torch.tensor(actions).float().to(device)    
    rewards = torch.tensor(rewards).float().to(device)    
    last_obs = torch.tensor(last_obs).float().to(device)    
    dones = torch.tensor(dones).float().to(device) 
    gammas = torch.tensor(gammas).float().to(device) 

    return obs, actions, rewards, last_obs, dones, gammas 

def experiences_to_numpy(experiences):
    '''
    Takes a batch of experiences: np.array[Experience] and transforms
    the experiences into separate np.arrays.

    Args:
    ----
    experiences: np.array[Experience, Experience, ...]
    
    Returns:
    -------
    states: np.array
    actions: np.array
    rewards: np.array
    new_states: np.array
    dones: np.array
    '''
    batch_size = len(experiences)
    states = np.array([experiences[i].state for i in range(batch_size)])
    actions = np.array([experiences[i].action for i in range(batch_size)])
    rewards = np.array([experiences[i].reward for i in range(batch_size)])
    new_states = np.array([experiences[i].new_state for i in range(batch_size)])
    dones = np.array([experiences[i].done for i in range(batch_size)])
    return (states, actions, rewards, new_states, dones)

def experiences_to_numpy_nstep(experiences):
    batch_size = len(experiences)
    obs = np.array(
            [exp.s_t.flatten() for exp in experiences]
            ).reshape((batch_size,-1))
    actions = np.array(
            [exp.a_t for exp in experiences]
            ).reshape((batch_size, -1))
    returns = np.array(
            [exp.disc_ret for exp in experiences]
            ).flatten()
    last_obs = np.array(
            [exp.s_tpn for exp in experiences]
            ).reshape((batch_size,-1))
    done_flags= np.array(
            [exp.done for exp in experiences]
            ).flatten()
    gammas = np.array(
            [exp.gamma_power for exp in experiences]
            ).flatten()
    return obs, actions, returns, last_obs, done_flags, gammas

def experiences_to_tensor(batch, device, include_gammas=False, batch_size=None):
    '''
    Takes a the arrays of SARSD and transforms them 
    to torch.tensors.

    Args: 
    ----
    batch: Tuple[np.array, np.array, ...] SARSD

    Returns:
    -------
    states: torch.tensor
    actions: torch.tensor
    rewards: torch.tensor
    new_states: torch.tensor
    dones: torch.tensor
    device: torch device available
    '''
    batch_size = len(batch) if batch_size == None else batch_size
    if include_gammas:
        states, actions, rewards, new_states, dones, gammas = batch
        gammas = torch.tensor(gammas).float().to(device).flatten()
    else:
        states, actions, rewards, new_states, dones = batch
    states = torch.tensor(states).float().to(device).reshape(batch_size,-1)
    actions = torch.tensor(actions).float().to(device).reshape(batch_size, -1) 
    rewards = torch.tensor(rewards).float().to(device).flatten()
    new_states = torch.tensor(new_states).float().to(device).reshape(batch_size, -1)
    dones = torch.tensor(dones).float().to(device).flatten()
    if include_gammas:
        return (states, actions, rewards, new_states, dones, gammas)
    else: 
        return (states, actions, rewards, new_states, dones)
   
def warm_up(agent, env, n_actions):
    done = True
    print('Agent: Warming up')
    for ia in tqdm(range(n_actions)):
        if done:
            state = env.reset()
            done = False
        action = env.action_space.sample()
        new_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, new_state, done)
        #env.render()
        state = new_state

    

def random_experience(state_dimension: int, action_dimension: int):
    '''
    Create a complete random experience, JUST FOR TESTING.

    Args:
    ----
    state_dimension
    action_dimension

    Returns:
    --------
    Experience(S,A,R,S,D)
    '''
    state = np.random.random(state_dimension)
    action = np.random.random(action_dimension)
    reward = np.random.random()
    new_state = np.random.random(state_dimension)
    done = True if np.random.random() > 0.5 else False
    exp = experience(state, action, reward, new_state, done)
    return exp


def episode_evaluation(
        agent, 
        agent_config,
        env_config=None,
        render=True):

    if env_config is not None:
        env = gym.make(agent_config.env.name, env_config=env_config)
    else:
        env = gym.make(agent_config.env.name)
    
    agent.actor.eval()
    state = env.reset()

    done=False

    total_reward = []
    ep_len=0
    states = []
    actions = []

    while not done:
        tensor_state = torch.tensor(state.astype(np.float32)).reshape(1,len(state))
        action = agent.actor(tensor_state)
        new_state, reward, done, info = env.step(action.cpu().detach().numpy().flatten())

        total_reward.append(reward)
        states.append(state)
        actions.append(action.cpu().detach().numpy())

        ep_len += 1
        if render:
            env.render()

        state=new_state
    agent.actor.train() 
    return total_reward, ep_len, states, actions

def exp_to_numpy(s_t,a_t,r_tp1,d_tp1,q_t,G_1):
    s_t = s_t.detach().cpu().numpy()
    a_t = a_t.detach().cpu().numpy()
    r_tp1 = r_tp1.detach().cpu().numpy()[0]
    d_tp1 = bool(d_tp1.detach().cpu()) 
    q_t = q_t.detach().cpu().numpy()[0]
    G_1 = G_1.detach().cpu().numpy()[0]
    return s_t,a_t,r_tp1,d_tp1,q_t,G_1

