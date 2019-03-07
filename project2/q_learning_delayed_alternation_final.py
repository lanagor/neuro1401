# Neuro 1401
# Spring 2019
# Group 6: Project 2

import numpy as np
import matplotlib.pyplot as plt 

"""Replicates Figure 2 in the Wilson paper."""

# Define q-learning parameters 
q_learning_params = {
                    'alpha': 0.03, 
                    'beta': 3, 
                    'reward': 1
                    }

max_iters = 500

# Set-up for each experimental condition
sham_lesioned = {'1': {'left': 0, 'right': q_learning_params['reward']}, 
                 '2': {'left': q_learning_params['reward'], 'right': 0}}
OFC_lesioned = {'1': {'left': 0, 'right': q_learning_params['reward']}}

def plot_alternation_correctness(sham_fraction_correct, OFC_fraction_correct, max_iters):
    
    bar_width = 0.2
    opacity = 0.8
     
    plt.bar(1, sham_fraction_correct, bar_width,
    color='b',
    label='Sham Lesions')
     
    plt.bar(1.5, OFC_fraction_correct, bar_width,
    color='r',
    label='OFC Lesions')
     
    plt.ylabel('Fraction Correct over {} Iterations'.format(max_iters))
    plt.title('Delayed Alternation Performance')
    plt.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        labelbottom=False)
    plt.legend()
     
    plt.tight_layout()
    plt.show()

def select_action(q_learning_params, q_values, state):
    '''Select an action given the current state and using the Luce rule.'''
    # Compute the probability of choosing 'right'
    weighted_q_values = np.zeros(2)
    actions = ['left', 'right']
    for action_index in range(2):
        weighted_q_values[action_index] = q_learning_params['beta'] * q_values[str(state)][actions[action_index]]
    prob_right = ((1.0 * np.exp(weighted_q_values[1])) / (1.0 * np.sum(np.exp(weighted_q_values))))
    # Choose an action based on the probability of choosing 'right' / 'left'
    return np.random.choice(['left', 'right'], p=[1 - prob_right, prob_right])

def q_update(state_action_reward_dict, q_values, q_learning_params, state, action):
    '''Given the most recently taken action (and the resulting reward), update the stored q values.'''
    q_values[str(state)][action] += q_learning_params['alpha'] * (state_action_reward_dict[str(state)][action] - q_values[str(state)][action])
    return q_values 

def check_correct(state_action_reward_dict, state, action):
    '''Checks whether the action selected yielded the largest possible reward.'''
    correct = False
    if len(state_action_reward_dict) == 2 and state_action_reward_dict[str(state)][action] > 0:
        correct = True
    if len(state_action_reward_dict) == 1 and state_action_reward_dict['1'][action] > 0:
        correct = True 
    return correct 

def delayed_alternation_learning(state_action_reward_dict, q_learning_params, max_iters):
    '''Executes the delayed alternation task with a simulated agent.'''
    if len(state_action_reward_dict) == 2:
        q_values = {'1': {'left': 0, 'right': 0}, 
                '2': {'left': 0, 'right': 0}}
    else: 
        q_values = {'1': {'left': 0, 'right': 0}}
    state = 1
    previous_action = None
    num_correct = 0.0
    for _ in range(max_iters):
        # Let the action rewarded be the opposite of the previous action, irrespective of the state
        if previous_action is not None and len(state_action_reward_dict) == 2:
            if previous_action == 'right':
                state_action_reward_dict[str(state)]['left'] = q_learning_params['reward']
                state_action_reward_dict[str(state)]['right'] = 0
            elif previous_action == 'left':
                state_action_reward_dict[str(state)]['left'] == 0
                state_action_reward_dict[str(state)]['right'] == q_learning_params['reward']
        elif previous_action is not None and len(state_action_reward_dict) == 1:
            if previous_action == 'right':
                state_action_reward_dict['1']['left'] = q_learning_params['reward']
                state_action_reward_dict['1']['right'] = 0
            elif previous_action == 'left':
                state_action_reward_dict['1']['left'] = 0
                state_action_reward_dict['1']['right'] = q_learning_params['reward']
        # Select an action
        action = select_action(q_learning_params, q_values, state)
        # Update the q-values 
        q_values = q_update(state_action_reward_dict, q_values, q_learning_params, state, action)
        # Check if the action taken resulted in the highest possible reward, and if so, increment num_correct
        correct = check_correct(state_action_reward_dict, state, action)
        if correct:
            num_correct += 1.0
        # In the sham lesioned case, update the state based on the action taken (associate 1 with left, 2 with right)
        if len(state_action_reward_dict) == 2:
            if action == 'left':
                state = 1
            elif action == 'right':
                state = 2
        # Update the previous action
        previous_action = action
    fraction_correct = (num_correct / (1.0 * max_iters))
    return fraction_correct


sham_fraction_correct = delayed_alternation_learning(sham_lesioned, q_learning_params, max_iters)
print("Sham Lesions -- Fraction Correct in Delayed Alternation: {}".format(sham_fraction_correct))

OFC_fraction_correct = delayed_alternation_learning(OFC_lesioned, q_learning_params, max_iters)
print("OFC Lesions -- Fraction Correct in Delayed Alternation: {}".format(OFC_fraction_correct))

plot_alternation_correctness(sham_fraction_correct, OFC_fraction_correct, max_iters)

