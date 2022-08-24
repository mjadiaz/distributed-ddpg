from collections import deque 
import ray 
import gym 
import numpy as np 
import torch 

from src.networks import Actor, Critic
from src.exploration import NOISES
from src.memory import nstep_experience
from src.utils import experiences_to_tensor
from src.utils import exp_to_numpy

@ray.remote
class Worker: 
    '''
    The actor colects experiences from its environment given an exploratory policy.
    The rate variable esp is external and constant. The actor holds the experiences
    in an internal local buffer before send it to the general replau buffer. 
    The actor updates its weights periodically.
    
    Adapted from: Enes Bilgin, Mastering Reinforcement Learning with Python.

    Args:
    -----
    actor_id
    replay_buffer
    parameter_server
    config
    eps
    eval = False
    env_config = None
    '''
    def __init__(self,
            actor_id,
            replay_buffer,
            parameter_server,
            config,
            eps,
            eval = False,
            env_config = None,
            ):
        
        self.actor_id = actor_id
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server
        self.config = config
        self.env_config = env_config
        self.prioritized = config.agent.prioritized
        
        self.gamma = self.config.agent.gamma
        self.eps = eps
        if config.agent.split_sigma:
            self.config.noise.sigma = config.noise.sigma*self.eps
        self.eval = eval

        # Initialize networks
        self.actor = Actor(self.config, worker=True)
        if self.prioritized:
            self.critic = Critic(self.config, worker=True)
            self.target_actor = Actor(self.config, target=True,worker=True)
            self.target_critic= Critic(self.config, target=True,worker=True)

        self.device = self.actor.device

        if self.env_config is None:
            self.env = gym.make(self.config.env.name)
        else: 
            self.env = gym.make(self.config.env.name, env_config=env_config)

        self.local_buffer = []
        self.state_dimension = self.config.env.state_dimension
        self.action_dimension = self.config.env.action_dimension
        
        # Add to agent config
        self.multi_step_n = self.config.agent.multi_step_n
        self.q_update_freq = self.config.agent.q_update_freq
        self.send_experience_freq =\
                self.config.agent.send_experience_freq

        self.continue_sampling = True
        self.current_episodes = 0
        self.current_steps = 0
       
        # Initialize noise
        self.noise = NOISES[self.config.noise.name](
                config=self.config.noise,
                size = self.config.env.action_dimension
                )
    
    def update_networks(self):
        if self.prioritized:
            self.update_networks_prioritized()
        else:
            self.update_networks_uniform()

    def update_networks_prioritized(self):
        if self.eval:
            pid_weights =\
                    self.parameter_server.get_eval_weights.remote()
        else:
            pid_weights =\
                    self.parameter_server.get_weights.remote()

        new_weights = ray.get(pid_weights)
        if new_weights[0]:
            actor_w, t_actor_w, critic_w, t_critic_w = new_weights
            self.actor.set_weights(actor_w)
            self.target_actor.set_weights(t_actor_w)
            self.critic.set_weights(critic_w)
            self.target_critic.set_weights(t_critic_w)

    def update_networks_uniform(self):
        if self.eval:
            pid_weights =\
                    self.parameter_server.get_eval_weights.remote(single='actor')
        else:
            pid_weights =\
                    self.parameter_server.get_weights.remote(single='actor')

        new_weights = ray.get(pid_weights)
        if new_weights:
            actor_w = new_weights
            self.actor.set_weights(actor_w)

    def update_networks_uniform_(self):
        '''
        Pull the weights from the parameter server.
        '''
        if self.eval:
            pid_actor =\
                self.parameter_server.get_eval_actor_weights.remote()
        else:
            pid_actor =\
                self.parameter_server.get_actor_weights.remote()

        if self.prioritized:
            pid_critic =\
                self.parameter_server.get_eval_critic_weights.remote()
            new_weights_critic = ray.get(pid_critic)

        new_weights_actor = ray.get(pid_actor)
        if new_weights_actor:
            self.actor.set_weights(new_weights_actor)
        if self.prioritized:
            if new_weights_critic:
                self.critic.set_weights(new_weights_critic)
        else:
            print(f'Actor-{self.actor_id}: Weights are not available yet, skipping.')

    @torch.no_grad()
    def get_action(self, state: np.ndarray):
        '''
        Forwards the state through the actor model to get
        the action prediction
        
        Args:
        ----
        state: np.ndarray

        Returns:
        -------
        action
        '''
        self.actor.eval()
        state = torch.tensor(state).float().to(self.device)#.reshape(1,-1)
        action = self.actor(state) +\
                torch.tensor(self.noise()).float().to(self.device)
                
        action = np.clip(   
                action.cpu().detach().numpy(), 
                self.config.env.action_min,
                self.config.env.action_max
                )
        self.actor.train()
        return action
    
    def get_n_step_trans_prioritized(self, n_step_buffer):
        '''
        Calculate the n-step discounted return 
        with the n_step_buffer.
        '''
        discounted_return = 0
        power_gamma = 1
        for transition in list(n_step_buffer)[:-1]:
            _, _, r_tp1, _,  _, G_1 = transition
            #G_t_n += G_1 
            discounted_return += power_gamma * r_tp1
            power_gamma *= self.gamma
        s_t, a_t, _, _ , q_t, _ = n_step_buffer[0]
        s_tpn, _, _, done, q_tpn, G_1   = n_step_buffer[-1]
        #print(f'state {s_t}, action {a_t}')
        n_step_td_error = discounted_return + power_gamma*q_tpn - q_t
        #n_step_td_error = q_tpn - q_t
        n_step_exp = nstep_experience(
                s_t, 
                a_t,
                discounted_return,
                s_tpn,
                done,
                power_gamma,
                )
        return n_step_exp, n_step_td_error 

    def get_n_step_trans_uniform(self, n_step_buffer):
        '''
        Calculate the n-step discounted return 
        with the n_step_buffer.
        '''
        discounted_return = 0
        power_gamma = 1
        for transition in list(n_step_buffer)[:-1]:
            _, _, r_tp1, _ = transition
            discounted_return += power_gamma * r_tp1
            power_gamma *= self.gamma
        s_t, a_t, _, _ = n_step_buffer[0]
        s_tpn, _, _, done   = n_step_buffer[-1]

        experience = (
                s_t, a_t, discounted_return, s_tpn, done, power_gamma
                )
        return experience

    def stop(self):
        self.continue_sampling = False
    
    def sample_uniform(self):
        print_state = 'eval' if self.eval else 'collector'
        print("Starting sampling in {}-actor {}".format(
            print_state,self.actor_id))

        self.update_networks()
        s_t = self.env.reset()
        episode_reward = 0 
        episode_length = 0
        n_step_buffer = deque(maxlen=self.multi_step_n + 1)
        
        while self.continue_sampling:
            a_t = self.get_action(s_t)
            s_tp1, r_tp1, done, info= self.env.step(a_t)
            n_step_buffer.append((s_t, a_t, r_tp1, done))

            if len(n_step_buffer) == self.multi_step_n + 1:
                n_step_exp =\
                        self.get_n_step_trans_uniform(n_step_buffer)
                self.local_buffer.append(n_step_exp)

            self.current_steps += 1
            episode_reward += r_tp1
            episode_length += 1
            if done:
                if self.eval:
                    break
                s_tp1 = self.env.reset()
                if len(n_step_buffer) > 1:
                    n_step_exp =\
                            self.get_n_step_trans_uniform(n_step_buffer)
                    self.local_buffer.append(n_step_exp)
                self.current_episodes += 1
                episode_reward = 0
                episode_length = 0
            s_t = s_tp1
            if self.current_steps %\
                    self.send_experience_freq == 0 and not self.eval:
                        self.send_experience_to_replay()
            if self.current_steps % \
                    self.q_update_freq == 0 and not self.eval:
                        self.update_networks()
        return episode_reward, episode_length

    def sample_prioritized(self):
        print_state = 'eval' if self.eval else 'collector'
        print("Starting sampling in {}-actor {}".format(
            print_state,self.actor_id))
        self.update_networks()
        s_t = self.env.reset()
        episode_reward = 0 
        episode_length = 0
        n_step_buffer = deque(maxlen=self.multi_step_n + 1)
        
        while self.continue_sampling:
            a_t = self.get_action(s_t)
            s_tp1, r_tp1, done, info= self.env.step(a_t)
            
            # We need to convert to tensors, to use the critic
            # and calculate the td-errors
            s_t, a_t, r_tp1, s_tp1, d_tp1 =\
                    experiences_to_tensor(
                            (s_t, a_t, r_tp1, s_tp1, done),
                            device=self.device,
                            batch_size=1
                            ) 

            self.critic.eval()
            self.target_actor.eval()
            self.target_critic.eval()

            ones = torch.ones_like(d_tp1).float().to(self.device)
   
            mu_tp1 = self.target_actor(s_tp1)
            
            
            target_q_tp1 = self.target_critic(
                    s_tp1, 
                    mu_tp1
                    ).flatten()

            y = r_tp1 + self.gamma * target_q_tp1 * (ones-d_tp1)

            q_t = self.critic(s_t, a_t)
           
            # Convert them back to numpy
            s_t,a_t,r_tp1,d_tp1,q_t,y =\
                    exp_to_numpy(s_t,a_t,r_tp1,d_tp1,q_t,y)
            n_step_buffer.append((s_t,a_t,r_tp1,d_tp1,q_t,y))

            if len(n_step_buffer) == self.multi_step_n + 1:
                if self.prioritized:
                    n_step_exp, n_step_td_error =\
                        self.get_n_step_trans_prioritized(n_step_buffer)
                    self.local_buffer.append(
                            (n_step_td_error, n_step_exp)       
                            )
                else:
                    n_step_exp =\
                            self.get_n_step_trans_uniform(n_step_buffer)
                    self.local_buffer.append(n_step_exp)

            self.current_steps += 1
            episode_reward += r_tp1
            episode_length += 1
            if done:
                if self.eval:
                    break
                s_tp1 = self.env.reset()
                if len(n_step_buffer) > 1:
                    if self.prioritized:
                        n_step_exp, n_step_td_error =\
                            self.get_n_step_trans_prioritized(n_step_buffer)
                        self.local_buffer.append(
                                (n_step_td_error, n_step_exp)       
                                )
                    else:
                        n_step_exp =\
                                self.get_n_step_trans_uniform(n_step_buffer)
                        self.local_buffer.append(n_step_exp)
                self.current_episodes += 1
                episode_reward = 0
                episode_length = 0
            if torch.is_tensor(s_tp1):
                s_tp1 = s_tp1.detach().cpu().numpy()
            s_t = s_tp1
            if self.current_steps %\
                    self.send_experience_freq == 0 and not self.eval:
                        self.send_experience_to_replay()
            if self.current_steps % \
                    self.q_update_freq == 0 and not self.eval:
                        self.update_networks()
        return episode_reward, episode_length
    
    def sample(self):
        if self.prioritized: 
            ep_return, ep_length = self.sample_prioritized()
        else:
            ep_return, ep_length = self.sample_uniform()
        return ep_return, ep_length

    def send_experience_to_replay(self):
        if self.prioritized:
            buffer = np.array(self.local_buffer, dtype=object)
            rf = self.replay_buffer.add_with_priorities.remote(buffer)
        else:
            rf = self.replay_buffer.add.remote(self.local_buffer)
        ray.wait([rf])
        self.local_buffer = []

