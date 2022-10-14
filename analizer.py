from src.analysis import QNetwork, PolicyNetwork
from train import agent_config, env_config, run_name 
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import softmax
import time
import imageio

from PIL import Image, ImageDraw
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_trajectories(actions_p):
    # Plots for toy models
    fig, ax = plt.subplots(figsize = (7,7))
    
    ax.title.set_text('Actor Policy Network')
    cp1 = ax.scatter(actions_p[:,0], actions_p[:,1], s=1, cmap='Blues', label='States')
    ax.legend(fontsize='xx-small',loc='upper center', bbox_to_anchor=(0.5,-0.07),
              ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return fig

def QA_Sampling(state=None, n_samples = 20, n_best=100):
    q_network.samples = n_samples
    if state is None:
        q_values = q_network.random_state_q_values()
        state = q_network.current_states[0] 
    else:
        q_values = q_network.get_q_values(state) 
        state = state

    actions = q_network.current_actions
    actions_q_values = np.hstack([actions,q_values])

    # Selection 
    # Sort from soft_max_q
    actions_q_values = actions_q_values[actions_q_values[:, 2].argsort()]
    # desending order and select first n_best
    actions_q_values = np.flip(actions_q_values, axis=0)[:n_best]
    # Calculate soft max
    action_index = np.arange(len(actions_q_values)).reshape(-1,1)
    soft_max_q = softmax(actions_q_values[:,2])
    actions_q_values = np.hstack([actions_q_values, soft_max_q.reshape(-1,1), action_index.reshape(-1,1) ])
    # Define new 100 new actions, index and softmax and q values
    soft_max_q =actions_q_values[:,3] 
    action_index = actions_q_values[:,4]
    q_values = actions_q_values[:,2]
    actions = actions_q_values[:,0:2]
    return state, actions, q_values, action_index, soft_max_q

def start_trajectories(steps, read=True):
    if read:

        st.video('gameplay.mp4', format="video/mp4", start_time=0)
    else:
        images = []
        state_space, _ = p_network.get_action_space(samples=100)
        fig = plot_trajectories(state_space)
        img = fig2img(fig)
        images.append(img)
        for n in range(steps): 
            # Update progress bar.
            _, actions_p = p_network.get_action_space(samples=100, state_space=state_space)
            fig = plot_trajectories(actions_p)
            img = fig2img(fig)
            images.append(img)
            state_space = actions_p
        imageio.mimwrite('gameplay.mp4', images, fps=5, quality=10)
        st.video('gameplay.mp4', format="video/mp4", start_time=0)


tab1, tab2, tab3, tab4 = st.tabs(["Q-Network", "Policy", 'Combined sampling', 'QA Sampling'])


q_network = QNetwork(
        agent_config,
        env_config,
        run_name,
        checkpoint_name = None,
        samples = 350
        )

p_network = PolicyNetwork(
        agent_config,
        env_config,
        run_name,
        checkpoint_name = None,
        samples = 50
        )
with tab1:
    q_values = q_network.random_state_q_values()
    state = q_network.current_states[0] 
    
    
    soft_max_q = softmax(q_values)
    actions = q_network.current_actions
    actions_q_values = np.hstack([actions,q_values, soft_max_q])
    max_q_index = np.argmax(soft_max_q)
    
    policy_next_state = p_network.get_action(state)
    
    # Plots for toy models
    fig, ax = plt.subplots(figsize = (4,4))
    
    ax.title.set_text('Critic (Q network)')
    cp1 = ax.scatter(actions[:,0], actions[:,1], c=soft_max_q, s=4, cmap='Blues', label='space of posible actions')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb1 = fig.colorbar(cp1, cax=cax, orientation='vertical')
    ax.scatter(state[0], state[1], c='Green',s=20, label='Initial Random state')
    ax.scatter(actions[max_q_index,0], actions[max_q_index,1], c='Orange',s=20, label='State with Max Q')
    ax.scatter(policy_next_state[0], policy_next_state[1],c='Cyan',s=20, label='Next State from Policy')
    ax.legend(fontsize='xx-small',loc='upper center', bbox_to_anchor=(0.5,-0.07),
              ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    
    
    text = '''
    # Q-network analysis
    Loads the saved Q-network (critic) model from the checkpoint specified in `train.py` script.
    For a random state in white, the plot shows all the possible actions with their corresponding Q-value.
    
    Reload the page to check a new random state.
    '''
    
    st.markdown( text)
    st.pyplot(fig)
with tab2:

    st.markdown('# Policy analysis')
    
    state_space, actions_p = p_network.get_action_space(samples=100)
    
    start_trajectories(100)
    
with tab3:
    q_network.samples = 100

    q_values = q_network.random_state_q_values()
    state = q_network.current_states[0] 
    
    
    actions = q_network.current_actions
    actions_q_values = np.hstack([actions,q_values])

    # Selection 
    # Sort from soft_max_q
    actions_q_values = actions_q_values[actions_q_values[:, 2].argsort()]
    # desending order and select first 100
    actions_q_values = np.flip(actions_q_values, axis=0)[:100]
    # Calculate soft max
    action_index = np.arange(len(actions_q_values)).reshape(-1,1)
    soft_max_q = softmax(actions_q_values[:,2])
    actions_q_values = np.hstack([actions_q_values, soft_max_q.reshape(-1,1), action_index.reshape(-1,1) ])
    # Define new 100 new actions, index and softmax and q values
    soft_max_q =actions_q_values[:,3] 
    action_index = actions_q_values[:,4]
    q_values = actions_q_values[:,2]
    actions = actions_q_values[:,0:2]

    max_q_index = np.argmax(soft_max_q)
    policy_next_state = p_network.get_action(state)

    
    fig, ax = plt.subplots(figsize = (4,4))
    
    ax.title.set_text('Q-A Sampling')
    cp1 = ax.scatter(actions_q_values[:,0], actions_q_values[:,1], c=actions_q_values[:,3], s=4, cmap='Blues', label='100 Best Q-Value proposal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb1 = fig.colorbar(cp1, cax=cax, orientation='vertical')

    ax.scatter(actions[max_q_index,0], actions[max_q_index,1], c='Orange',s=20, label='State with Max Q')
    ax.scatter(policy_next_state[0], policy_next_state[1],c='Cyan',s=20, label='Next State from Policy')
    ax.scatter(state[0], state[1], c='Green',s=20, label='Initial Random state')
    ax.legend(fontsize='xx-small',loc='upper center', bbox_to_anchor=(0.5,-0.07),
              ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    st.pyplot(fig)
    
    
    # Second Plot
    q_random_choice = np.random.choice(action_index, replace=False, p=soft_max_q.flatten())
    q_next_state_choice = actions[int(q_random_choice)]

    policy_corrected = p_network.get_action(q_next_state_choice)
    fig, ax = plt.subplots(figsize = (4,4))
    
    ax.title.set_text('Q-A Sampling')

    ax.scatter(q_next_state_choice[0], q_next_state_choice[1], c='Green',s=20, label='Sampled point from Best 100 Q')
    ax.scatter(policy_corrected[0], policy_corrected[1], c='Cyan',s=20, label='Policy transformed state')
    ax.legend(fontsize='xx-small',loc='upper center', bbox_to_anchor=(0.5,-0.07),
              ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    st.pyplot(fig)

# Third plot
    q_values = q_network.get_q_values(policy_corrected)
    state = policy_corrected
    
    
    actions = q_network.current_actions
    actions_q_values = np.hstack([actions,q_values])

    # Selection 
    # Sort from soft_max_q
    actions_q_values = actions_q_values[actions_q_values[:, 2].argsort()]
    # desending order and select first 100
    actions_q_values = np.flip(actions_q_values, axis=0)[:100]
    # Calculate soft max
    action_index = np.arange(len(actions_q_values)).reshape(-1,1)
    soft_max_q = softmax(actions_q_values[:,2])
    actions_q_values = np.hstack([actions_q_values, soft_max_q.reshape(-1,1), action_index.reshape(-1,1) ])
    # Define new 100 new actions, index and softmax and q values
    soft_max_q =actions_q_values[:,3] 
    action_index = actions_q_values[:,4]
    q_values = actions_q_values[:,2]
    actions = actions_q_values[:,0:2]

    max_q_index = np.argmax(soft_max_q)
    policy_next_state = p_network.get_action(state)

    
    fig, ax = plt.subplots(figsize = (4,4))
    
    ax.title.set_text('Q-A Sampling')
    cp1 = ax.scatter(actions[:,0], actions[:,1], c=actions_q_values[:,3], s=4, cmap='Blues', label='100 Best Q-Value proposal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb1 = fig.colorbar(cp1, cax=cax, orientation='vertical')

    ax.scatter(actions[max_q_index,0], actions[max_q_index,1], c='Orange',s=20, label='State with Max Q')
    ax.scatter(policy_next_state[0], policy_next_state[1],c='Cyan',s=20, label='Next State from Policy')
    ax.scatter(state[0], state[1], c='Green',s=20, label='Initial Random state')
    ax.legend(fontsize='xx-small',loc='upper center', bbox_to_anchor=(0.5,-0.07),
              ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    st.pyplot(fig)
    
def qa_plot(state, actions, q_values, action_index, soft_max_q, policy_next_state, q_next_state_choice):

    fig, ax = plt.subplots(figsize = (6,6))
    
    ax.title.set_text('Q-A Sampling')
    cp1 = ax.scatter(actions[:,0], actions[:,1], c=soft_max_q, s=4, cmap='Blues', label='100 Best Q-Value proposal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb1 = fig.colorbar(cp1, cax=cax, orientation='vertical')
    ax.scatter(q_next_state_choice[0], q_next_state_choice[1], c='Orange',s=20, label='Q-State-Sampling')
    ax.scatter(policy_next_state[0], policy_next_state[1],c='Cyan',s=20, label='Policy correction to Q-State-Sampling')
    ax.scatter(state[0], state[1], c='Green',s=20, label='Current State')
    ax.legend(fontsize='xx-small',loc='upper center', bbox_to_anchor=(0.5,-0.07),
              ncol=2, fancybox=True, shadow=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return fig
with tab4:
    # total_samples = n_samples ** 2
    n_samples, n_best = 60, 100
    state, actions, q_values, action_index, soft_max_q = QA_Sampling(n_samples=n_samples, n_best=n_best)
    images = [] 
    for n in range(30):

        state , actions, q_values, action_index, soft_max_q = QA_Sampling(state, n_samples=n_samples, n_best=n_best)
        q_random_choice = np.random.choice(action_index, replace=False, p=soft_max_q.flatten())
        q_next_state_choice = actions[int(q_random_choice)]

        policy_corrected = p_network.get_action(q_next_state_choice)

    
        fig = qa_plot(state, actions, q_values, action_index, soft_max_q, policy_corrected, q_next_state_choice)
        
        img = fig2img(fig)
        images.append(img)

        state = policy_corrected

    imageio.mimwrite('qasampling.mp4', images, fps=5, quality=10)
    st.video('qasampling.mp4', format="video/mp4", start_time=0)


