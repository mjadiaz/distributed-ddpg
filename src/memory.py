import numpy as np
import ray

from omegaconf import DictConfig
from collections import deque, namedtuple


experience = namedtuple(
        'Experience',
        field_names=[
            'state',
            'action',
            'reward',
            'new_state',
            'done'] 
        )

nstep_experience = namedtuple(
        'n_step_experience',
        field_names=[
            's_t',
            'a_t',
            'disc_ret',
            's_tpn',
            'done',
            'gamma_power',
            ]
        )
@ray.remote
class ExperienceReplayMemory:
    '''Vanilla Experience Replay Memory for storing past experiences'''
    def __init__(self, config: DictConfig):
        self.config = config
        
        self.batch_size = self.config.batch_size
        self.max_size = int(self.config.max_size)
        self.memory = deque(maxlen=int(self.max_size))
        
        self.samples_counter = 0

    def __len__(self):
        return len(self.memory)

    def add(self, experience):
        for e in experience:
            self.memory.append(e)
            self.samples_counter += 1

    def sample(self):
        sample_ix = np.random.randint(
                len(self.memory),
                size=self.batch_size
                )
        return [self.memory[ix] for ix in sample_ix]
        

    def get_total_samples(self):
        return self.samples_counter


        
# Adapted From: https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_10/chapter-10.ipynb
@ray.remote
class PrioritizedReplayBuffer:
    def __init__(self, config: DictConfig):
        self.config = config
        self.max_size =  int(config.max_size)
        self.batch_size = int(config.batch_size)

        # Add to base-ERM config 
        # if not rank based, then proportional
        self.rank_based = config.rank_based 
        # How much prioritization to use. 0 is uniform (nopririty)
        # 1 is full priority
        self.alpha = config.alpha
        # Beta is the bias correction, 0 is no correction
        # 1 is full correction.
        # beta0 is the initial value.
        self.beta = config.beta
        self.beta0 = config.beta
        # Decreasing beta rate.
        self.beta_rate = config.beta_rate

        self.memory = np.empty((self.max_size, 2), dtype=np.ndarray)
        
        self.samples_counter = 0
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self._eps = 1e-4#  Small epsilon constant to avoid small priorities

    def update(self, idxs, td_errors):
        '''
        Sort the memory according to the td error magnitudes
        in decreasing order
        '''
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based:
            sorted_arg = self.memory[:self.n_entries, self.td_error_index]\
                    .argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def add_with_priorities(self,buffer):
        '''
        Apex variant for adding experiences to the memory.
        The argument buffer is np.dnarray([[Tuple, float],...]) 
        where Tuple is the experience Tuple and float is the td_error.

        Adds the local buffer of an actor, with initial 
        priorities or td errors.
        '''
        samples = buffer[:,self.sample_index]
        td_errors = buffer[:,self.td_error_index]
        n_samples = len(samples)
        indexes = np.arange(
                self.next_index, 
                self.next_index + n_samples
                ) % self.max_size

        self.memory[indexes, self.td_error_index] = np.abs(td_errors) 
        self.memory[indexes, self.sample_index] = samples
        self.n_entries = min(self.n_entries + n_samples, self.max_size)
        self.next_index = (indexes[-1] + 1) % self.max_size
        self.samples_counter += n_samples

    def add_with_priorities_(self, buffer):
        samples = buffer[:,self.sample_index]
        td_errors = buffer[:,self.td_error_index]
        n_samples = len(samples)
        
        for s, td in zip(samples, td_errors):
            self.memory[self.next_index, self.td_error_index] = np.abs(td)
            self.memory[self.next_index, self.sample_index] = s
            self.n_entries = min(self.n_entries +1, self.max_size)
            self.next_index += 1
            self.next_index = self.next_index % self.max_size
    



    def add(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[
                    :self.n_entries,
                    self.td_error_index
                    ].max()
        self.memory[self.next_index, self.td_error_index ] = np.array(priority)
        self.memory[self.next_index, self.sample_index] = np.array(sample)
        self.n_entries = min(self.n_entries +1, self.max_size)
        self.next_index += 1
        self.next_index = self.next_index % self.max_size

    def _update_beta(self):
        self.beta = min(1.0, self.beta*self.beta_rate**-1)
        return self.beta

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size == None else batch_size
        self._update_beta()
        entries = self.memory[:self.n_entries]

        if self.rank_based:
            priorities = 1/(np.arange(self.n_entries)+1)
        else: 
            priorities = entries[:, self.td_error_index] + self._eps
        
        scaled_priorities = priorities**self.alpha
        scaled_priorities = scaled_priorities.astype('float64')
        pri_sum = np.sum(scaled_priorities)
        norm_pri = scaled_priorities/pri_sum
        probs = np.array(
                norm_pri,
                dtype=np.float64
                )

        weights = (self.n_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()
        idxs = np.random.choice(
                self.n_entries,
                batch_size,
                replace=False,
                p=probs
                )
        samples = entries[idxs] 
        samples_stack = samples[:,self.sample_index].flatten()
        idxs_stack = idxs
        weights_stack = normalized_weights[idxs]
        return idxs_stack, weights_stack, samples_stack


    def __len__(self):
        return self.n_entries

    def get_total_samples(self):
        return self.samples_counter


        
