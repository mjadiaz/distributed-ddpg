# distributed-ddpg
# Contents
- [Training the Distributed](#Training the Distributed-DDPG)
	- [Config file](##Config file)
-  [Conda environment](#Conda environment)
	- [Working on toy-models branch](##Working on toy-models branch)
	- [Running on Mac M1](##Running on Mac M1)

--- 

# Training the Distributed-DDPG

To run the algorithm, configure the agent and environment config in `train.py` script and run it with `python train.py`.

Remember to edit the `RUN_NAME` variable, which is the name of the folder to save the trained agent.


## Config file

```python
APEX_DDPG_DEFAULT_CONFIG = {
        'agent': {
            'actor_lr': 0.001,
            'critic_lr': 0.001,
            'tau': 0.002,
            'gamma': 0.99,
            'save_path': 'results/ddpg', #  Is modified by the RUN_NAME variable
            'prioritized': False,		# Prioritized memory is not working
            'num_train_workers': 3,
            'num_eval_workers': 2,
            'max_exploration_eps': 1.0,
            'split_sigma': True,
            'learning_starts': 100,
            'max_samples': 5e6,		# To stop the training
            'timesteps_per_iteration': 1000,
            'multi_step_n': 3,		# N-step learning
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
            'name': 'Gaussian', # Ornstein–Uhlenbeck needs fixing
            'theta': 0.2,
            'sigma': 0.15, # Standard deviation for the Gaussian noise 
            'dt': 1e-3,
            'scale': 1.0,
            'epsilon': 0.3, # Epsilon controling greedy_sigma 
            'greedy_sigma': 0.0,
            'decrease': False,  # Decrease the sigma over timesteps
            'final_sigma': 0.02, # Final sigma for decreasing
            'initial_decreasing_step': 8000,
            'final_decreasing_step': 12000,
            },
        'running_on': 'M1'
        }

```
# Conda environment 
To use the package create a new conda environment with:

`conda env create --file environment.yml`

## Working on toy-models branch
- Clone following repo `git@github.com:mjadiaz/toy-models.git` and install the package on the toy-models folder with `pip install --editable .`. So that you can modify the environment.

- The `analizer.py` script displays a visualization for the Q-network (Critic). To use the `analizer.py` install [streamlit](https://streamlit.io/) with `pip install streamlit` and run the analizer script with:

`streamlit run analizer.py`

## Running on Mac M1

For running on Mac M1 with gpu acceleration update the pytorch version to:

`conda install pytorch -c pytorch-nightly`

and in the agent config turn `'running_on': 'M1'`. Also force reinstall with conda the following package,

`conda install grpcio --force`
