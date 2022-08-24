import torch
import torch.nn as nn
import torch.optim as optim
import os


from omegaconf import OmegaConf, DictConfig


def save_model(model, path=None):
    if path:
        new_path = os.path.join(model.save_path,path)
        model_file_path = os.path.join(
                new_path,
                model.name)
    else:
        new_path = model.save_path
        model_file_path = os.path.join(model.save_path, model.name)

    if not os.path.exists(new_path):
        os.makedirs(new_path)
    torch.save(model.get_weights(), model_file_path)

def load_model(model, save_path, checkpoint):
    model_file_path = os.path.join(save_path,checkpoint, model.name)
    model.load_state_dict(torch.load(model_file_path))



class Actor(nn.Module):
    '''
    Actor network for DDPG
    '''
    def __init__(self, config: DictConfig, target=False, worker=False):
        super().__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if worker: 
            self.device = 'cpu'

        self.name = 'actor'
        if target:
            self.name = 'target_actor'
        self.save_path = config.agent.save_path
        self.learning_rate = config.agent.actor_lr
        self.state_dimension = config.env.state_dimension   
        self.action_dimension = config.env.action_dimension

        self.linear_block = nn.Sequential(
                nn.Linear(self.state_dimension, 400),
                #nn.BatchNorm1d(400),
                nn.LayerNorm(400),
                nn.ReLU(),
                nn.Linear(400, 300),
                #nn.BatchNorm1d(300),
                nn.LayerNorm(300),
                nn.ReLU(),
                nn.Linear(300, self.action_dimension),
                nn.Tanh()
                )
        self.optimizer = optim.Adam(
                self.parameters(), lr=self.learning_rate
                )
        self.to(self.device)

    def forward(self, x):
        x = self.linear_block(x)
        return x
    
    def get_weights(self):
        return {k: v.cpu() for k,v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def save_model(self, path=None):
        save_model(self, path)

    def load_model(self, path, checkpoint):
        load_model(self, path, checkpoint)

class Critic(nn.Module):
    '''
    Critic network for DDPG
    '''
    def __init__(self, config: DictConfig, target=False, worker=False):
        super().__init__()
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if worker:
            self.device = 'cpu'

        self.name = 'critic'
        if target:
            self.name = 'target_critic'
        self.save_path = config.agent.save_path
        self.learning_rate = config.agent.critic_lr
        self.state_dimension = config.env.state_dimension   
        self.action_dimension = config.env.action_dimension

        self.batch_size = self.config.memory.batch_size

        self.linear_block = nn.Sequential(
                nn.Linear(self.state_dimension + self.action_dimension, 400),
                #nn.BatchNorm1d(400),
                nn.LayerNorm(400),
                nn.ReLU(),
                nn.Linear(400, 300),
                #nn.BatchNorm1d(300),
                nn.LayerNorm(300),
                nn.ReLU(),
                nn.Linear(300, 1),
                )
        self.optimizer = optim.Adam(
                self.parameters(), lr=self.learning_rate
                )
        self.to(self.device)

    def forward(self, x,y):
        if not(len(x.shape) == 1):
            q = self.linear_block(torch.cat([x,y],1))
            return q
        else:
            q = self.linear_block(torch.cat([x,y],0))
            return q
    
    def get_weights(self):
        return {k: v.cpu() for k,v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def save_model(self,path=None):
        save_model(self, path)

    def load_model(self, path, checkpoint):
        load_model(self, path, checkpoint)

# Adapted from: 
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter17/lib/model.py
class D4PGCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.obs_size = self.config.env.state_dimension
        self.act_size = self.config.env.action_dimension
        self.n_atoms = self.config.agent.n_atoms
        self.v_min = self.config.agent.v_min
        self.v_max = self.config.agent.v_max

        self.obs_net = nn.Sequential(
            nn.Linear(self.obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + self.act_size, 300),
            nn.ReLU(),
            nn.Linear(300, self.n_atoms)
        )

        delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.register_buffer("supports", torch.arange(
            self.v_min, self.v_max + self.delta, self.delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
