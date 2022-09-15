from src.analysis import QNetwork
from train import agent_config, env_config, run_name 
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

'''

'''

q_network = QNetwork(
        agent_config,
        env_config,
        run_name,
        checkpoint_name = None,
        samples = 200
        )
q_values = q_network.random_state_q_values()
state = q_network.current_states[0] 

actions = q_network.current_actions

# Plots for toy models
fig, ax = plt.subplots(figsize = (4,4))

ax.title.set_text('Critic (Q network)')
cp1 = ax.scatter(actions[:,0], actions[:,1], c=q_values, s=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cb1 = fig.colorbar(cp1, cax=cax, orientation='vertical')
ax.scatter(state[0], state[1], c='w',s=20)


text = '''
# Q-network analysis
Loads the saved Q-network (critic) model from the checkpoint specified in `train.py` script.
For a random state in white, the plot shows all the possible actions with their corresponding Q-value.

Reload the page to check a new random state.
'''

st.markdown( text)
st.pyplot(fig)
#st.sidebar.button('Restart')
