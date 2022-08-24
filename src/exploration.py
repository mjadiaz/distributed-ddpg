import numpy as np


class OrnsteinUhlenbeckNoise:
    def __init__(self, config, size, x0=None):
        self.theta = config.theta
        self.mu = size
        self.sigma = config.sigma
        self.dt = config.dt
        self.x0 = x0
        self.reset()    
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev)*self.dt +\
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
    

class GaussianNoise:
    def __init__(self, config, size):
        self.sigma = config.sigma
        self.final_sigma = config.final_sigma
        self.scale = config.scale
        self.size = size 
        self.epsilon = config.epsilon
        self.greedy_sigma = config.greedy_sigma
        self.decrease = config.decrease
        self.initial_step = config.initial_decreasing_step
        self.final_step = config.final_decreasing_step
        self.global_counter = 0
        self.d_counter = 0
        self.step = (self.sigma - self.final_sigma)/\
                (self.final_step - self.initial_step)

    def __call__(self):
        if self.decrease:
            in_range = (self.global_counter >= self.initial_step)\
                    and (self.global_counter <= self.final_step)
            if in_range:
                self.sigma += - self.step
        sigma = self.sigma
        if (1-self.epsilon) < np.random.uniform():
            sigma = self.greedy_sigma

        x = self.scale*np.random.normal(
                0, scale=sigma, size=self.size
                )
        self.global_counter +=1
        return x

    def reset(self):
        pass


NOISES = {
        'Gaussian': GaussianNoise,
        'OrnsteinUhlenbeck': OrnsteinUhlenbeckNoise}
