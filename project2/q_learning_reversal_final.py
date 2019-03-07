# Neuro 1401
# Spring 2019
# Group 6: Project 2

"""Note that in the original paper they average the results from five reversals. For simplicity, this script only executes 
one reversal, but (with some added variance) the results are the same as those from the paper. 

The plot produced at the end of the script replicates Figure 1 from the Wilson paper. 
"""

import numpy as np
import matplotlib.pyplot as plt 

# Define q-learning parameters 

q_learning_params = {
                    'alpha': 0.03, 
                    'beta': 3, 
                    'reward': 1
                    }

# Set-up for each experimental condition
sham_lesioned = {'1': {'left': 0, 'right': q_learning_params['reward']}, 
                 '2': {'left': q_learning_params['reward'], 'right': 0}}
OFC_lesioned = {'1': {'left': 0, 'right': q_learning_params['reward']}}

def plot_reversal_errors(sham_initial_errors, sham_reversal_errors, OFC_initial_errors, OFC_reversal_errors):
    conditions = 2
    initial_errors = (int(sham_initial_errors), int(OFC_initial_errors))
    reversal_errors = (int(sham_reversal_errors), int(OFC_reversal_errors))
    
    fig, ax = plt.subplots()
    index = np.arange(conditions)
    bar_width = 0.2
    opacity = 0.8
     
    plt.bar(index, initial_errors, bar_width,
    color='b',
    label='Sham Lesions')
     
    plt.bar(index + bar_width, reversal_errors, bar_width,
    color='r',
    label='OFC Lesions')
     
    plt.xlabel('Condition')
    plt.ylabel('Errors to Criterion')
    plt.title('Reversal Learning Performance')
    plt.xticks(index + 0.5 * bar_width, ('Before Reversal', 'After Reversal'))
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

def run_trial(state_action_reward_dict, q_learning_params, q_values, state):
    '''Let the simulated agent take actions until they reach 90% correctness, and return how long it took them.'''
    num_correct = 0.0
    trials = 1.0
    while (num_correct / trials) < 0.9:
        trials += 1.0
        action = select_action(q_learning_params, q_values, state)
        q_values = q_update(state_action_reward_dict, q_values, q_learning_params, state, action)
        # Conditions for the sham lesioned case
        if len(state_action_reward_dict) == 2 and state == 1:
            if action == 'right':
                num_correct += 1.0
        elif len(state_action_reward_dict) == 2 and state == 2:
            if action == 'left':
                num_correct += 1.0
        # Conditions for the OFC lesioned case 
        elif len(state_action_reward_dict) == 1 and state_action_reward_dict[str(state)]['right'] > 0:
            if action == 'right':
                num_correct += 1.0
        elif len(state_action_reward_dict) == 1 and state_action_reward_dict[str(state)]['left'] > 0:
            if action == 'left':
                num_correct += 1.0
    return num_correct, trials 

def reversal_learning(state_action_reward_dict, q_learning_params):
    '''Compute how many errors our simulated agent makes until they reach 90% accuracy in a reversal learning task.'''
    # Initialize q_values 
    if len(state_action_reward_dict) == 2:
        q_values = {'1': {'left': 0, 'right': 0}, 
                    '2': {'left': 0, 'right': 0}}
    else:
        q_values = {'1': {'left': 0, 'right': 0}}
    # Start in state 1
    state = 1
    # Take actions until 90% correct 
    initial_correct, initial_trials = run_trial(state_action_reward_dict, q_learning_params, q_values, state)  
    # Compute performance metric 
    initial_errors_to_criterion = initial_trials - initial_correct
    # Change state if sham lesioned
    if len(state_action_reward_dict) == 2:
        state = 2
    # Otherwise, change action receiving reward (since OFC lesioned)
    else:
        state_action_reward_dict['1']['left'] = q_learning_params['reward']
        state_action_reward_dict['1']['right'] = 0
    # Take actions until 90% correct
    reversal_correct, reversal_trials = run_trial(state_action_reward_dict, q_learning_params, q_values, state)
    # Compute performance metric 
    reversal_errors_to_criterion = reversal_trials - reversal_correct
    return initial_errors_to_criterion, reversal_errors_to_criterion 

# Run experiment with sham lesioned agent
sham_initial_errors, sham_reversal_errors = reversal_learning(sham_lesioned, q_learning_params)

# Run experiment with OFC lesioned agent 
OFC_initial_errors, OFC_reversal_errors = reversal_learning(OFC_lesioned, q_learning_params)

# Plot results for comparison with paper plots 
plot_reversal_errors(sham_initial_errors, sham_reversal_errors, OFC_initial_errors, OFC_reversal_errors)