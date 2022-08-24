import time
import ray
import numpy as np
import torch
import torch.nn.functional as F

from src.networks import Actor, Critic
from src.utils import experiences_to_numpy_nstep
from src.utils import experiences_to_tensor
from src.utils import batch_to_tensor

@ray.remote
class Learner:
    def __init__(self,
            config,
            replay_buffer,
            parameter_server
            ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server
        
        # Initialize networks
        self.actor = Actor(self.config)
        self.target_actor = Actor(self.config)
        self.critic = Critic(self.config)
        self.target_critic = Critic(self.config)

        # Hyperparmeters 
        self.gamma = config.agent.gamma
        self.tau = config.agent.tau 
        self.batch_size = config.memory.batch_size
        self.prioritized = config.agent.prioritized
        self.learning_starts = config.agent.learning_starts


        self.total_collected_samples = 0
        self.samples_since_last_update = 0

        self.send_weights()

        self.stopped = False

        self.update_target_networks()#tau=1.)
        self.device = self.actor.device

    def update_target_networks(self, tau: float = None):
        '''
        Updates the parameters of the target networks, tracking 
        the principal networks.

        tau=1 means complete copy.

        Args:
        ----
        tau: Slow tracking parameter << 1
        '''
        if tau == None:
            tau = self.tau

        def parameters_update(network, target_network, tau=tau):
            net_params = dict(network.named_parameters())
            target_net_params = dict(target_network.named_parameters())


            for name in net_params:  
                net_params[name] = tau*net_params[name].clone() \
                        + (1. - tau)*target_net_params[name].clone()

            target_network.load_state_dict(net_params)

        parameters_update(self.actor, self.target_actor)
        parameters_update(self.critic, self.target_critic)

    def send_weights_to_parameter_server(self):
        #self.parameter_server.update_actor_weights.remote(
        #        self.actor.get_weights()
        #        )
        #self.parameter_server.update_critic_weights.remote(
        #        self.critic.get_weights()
        #        )
        self.parameter_server.update_weights.remote(
                self.actor.get_weights(),
                self.target_actor.get_weights(),
                self.critic.get_weights(),
                self.target_critic.get_weights(),
                )

    def start_learning(self):
        print(f'learner: Learning starting.')
        self.send_weights()
        while not self.stopped:
            sid = self.replay_buffer.get_total_samples.remote()
            total_samples = ray.get(sid)

            if total_samples >= self.learning_starts:
                self.optimize()

    def optimize_prioritized(self):
        samples = ray.get(
                self.replay_buffer.sample.remote()
                )
        if samples: 
            if self.prioritized:
                idxs, weights, experience_batch = samples 
            else:
                experience_batch = samples

            N = len(experience_batch)
            self.total_collected_samples += N
            self.samples_since_last_update += N
            experiences = experiences_to_numpy_nstep(experience_batch)
            obs, actions, rewards, last_obs, dones, gammas =\
                    experiences_to_tensor(
                            experiences, 
                            self.device, 
                            include_gammas=True, 
                            batch_size=N
                            )
            # PER DDPG Learning
            self.target_actor.eval()
            self.target_critic.eval()
            self.critic.eval()

            mu_tpnp1 = self.target_actor(last_obs)
            target_q_tpnp1 = self.target_critic(last_obs, mu_tpnp1).flatten()
    
            ones = torch.ones(self.batch_size).to(self.device)
            y = rewards + gammas * target_q_tpnp1.flatten() *\
                    (ones - dones)
            values = self.critic(obs, actions).flatten()
            
            td_error = y - values

            self.critic.train()
            self.critic.optimizer.zero_grad()
            if self.prioritized:
                weights = torch.tensor(weights)\
                        .to(self.device).float().flatten()
                zeros = torch.zeros_like(td_error).to(self.device).float()
                critic_loss = F.mse_loss(weights*td_error,zeros)
            else:
                critic_loss = F.mse_loss(y,values)
            
            critic_loss.backward()
            self.critic.optimizer.step()

            td_error = td_error.detach().cpu().numpy().flatten()
            uid = self.replay_buffer.update.remote(idxs,td_error)
            ray.get(uid)
                

            self.critic.eval()
            self.actor.optimizer.zero_grad()

            mu = self.actor(obs)
            actor_loss = -self.critic(obs, mu)
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_target_networks()

            self.send_weights()
            return True
        else:
            print("No samples received from the buffer.")
            time.sleep(5)
            return False

    def optimize_uniform(self):
        samples = ray.get(
                self.replay_buffer.sample.remote()
                )
        if samples: 
            N = len(samples)
            self.total_collected_samples += N
            self.samples_since_last_update += N
            
            obs, actions, rewards, last_obs, dones, gammas =\
                    batch_to_tensor(samples, batch_size=N, device=self.device)


            # PER DDPG Learning
            self.target_actor.eval()
            self.target_critic.eval()
            self.critic.eval()

            target_actions = self.target_actor(last_obs)
            target_values = self.target_critic(last_obs, target_actions).flatten()
    
            ones = torch.ones(self.batch_size).to(self.device)

            y = rewards + gammas * target_values *(ones - dones)
            values = self.critic(obs, actions).flatten()
            
            self.critic.train()
            self.critic.optimizer.zero_grad()

            critic_loss = F.mse_loss(y,values)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.critic.eval()
            self.actor.optimizer.zero_grad()

            mu = self.actor(obs)
            actor_loss = -self.critic(obs, mu)
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_target_networks()

            self.send_weights()
            return True
        else:
            print("No samples received from the buffer.")
            time.sleep(5)
            return False

    def optimize(self):
        if self.prioritized:
            opt = self.optimize_prioritized()
        else:
            opt = self.optimize_uniform()
        return opt
            
    def send_weights(self):
        #aid = self.parameter_server.update_actor_weights.remote(
        #        self.actor.get_weights()
        #        )
        #cid = self.parameter_server.update_critic_weights.remote(
        #        self.critic.get_weights()
        #        ) 
        #ray.get(aid)
        #ray.get(cid)
        sid = self.parameter_server.update_weights.remote(
                self.actor.get_weights(),
                self.target_actor.get_weights(),
                self.critic.get_weights(),
                self.target_critic.get_weights(),
                )
        ray.get(sid)
    
    def stop(self):
        self.stopped = True


