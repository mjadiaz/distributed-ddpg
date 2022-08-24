import ray 
from src.networks import Actor, Critic
from omegaconf import DictConfig


class NetworkContainer:
    def __init__(self, network):
        self.weights = None
        self.eval_weights = None
        self.network = network

    def update_weights(self, new_parameters):
        self.weights = new_parameters
        return True

    def get_weights(self):
        return self.weights
    
    def get_eval_weights(self):
        return self.eval_weights

    def set_eval_weights(self):
        self.eval_weights = self.weights
        return True


    def save_eval_weights(
            self,
            path=None
            ):
        self.network.set_weights(self.eval_weights)
        self.network.save_model(path=path)
    
@ray.remote 
class NewParameterServer:
    def __init__(self, config:DictConfig):
        self.containers = dict(
                actor = NetworkContainer(Actor(config)),
                target_actor = NetworkContainer(Actor(config, target=True)),
                critic = NetworkContainer(Critic(config)),
                target_critic = NetworkContainer(Critic(config, target=True))
                )
        self.cont_names = self.containers.keys()

    def update_weights(self, 
            actor_params,
            target_actor_params,
            critic_params,
            target_critic_params
            ):
        new_params_list = [
                actor_params,
                target_actor_params,
                critic_params,
                target_critic_params
                ]
        for cont_name, new_params in zip(self.cont_names,new_params_list):
            self.containers[cont_name].update_weights(new_params)

    
    def get_weights(self, single=None):
        if not single:
            weights = [self.containers[name].get_weights()\
                    for name in self.cont_names]
        else:
            weights = self.containers[single].get_weights()
        return weights 

    def get_eval_weights(self, single=None):
        if not single:
            weights = [self.containers[name].get_eval_weights()\
                    for name in self.cont_names]
        else:
            weights = self.containers[single].get_eval_weights()
        return weights 

    def set_eval_weights(self):
        [self.containers[name].set_eval_weights()\
                for name in self.cont_names]
        return True

    def save_eval_weights(self,path=None):
        [self.containers[name].save_eval_weights(path=path)\
                for name in self.cont_names]


@ray.remote
class ParameterServer:
    def __init__(self, config: DictConfig):
        self.actor_weights = None
        self.eval_actor_weights = None

        self.critic_weights = None
        self.eval_critic_weights = None

        self.actor =  Actor(config)
        self.critic = Critic(config)

    def update_actor_weights(self, new_parameters):
        self.actor_weights = new_parameters
        return True
        
    def update_critic_weights(self, new_parameters):
        self.critic_weights = new_parameters
        return True

    def get_actor_weights(self):
        return self.actor_weights

    def get_critic_weights(self):
        return self.critic_weights

    def get_eval_critic_weights(self):
        return self.eval_critic_weights

    def get_eval_actor_weights(self):
        return self.eval_actor_weights

    def set_eval_weights(self):
        self.eval_actor_weights = self.actor_weights
        self.eval_critic_weights = self.critic_weights
        return True

    def save_eval_weights(self,
                          filename=
                          'checkpoints/model_checkpoint'):
        self.actor.set_weights(self.eval_actor_weights)
        self.actor.save_model()

        self.critic.set_weights(self.eval_critic_weights)
        self.critic.save_model()
        print("Saved.")


