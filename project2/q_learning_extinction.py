# Neuro 1401
# Spring 2019
# Group 6: Project 2

"""Note that in the original paper they average the results from five reversals. For simplicity, this script only executes 
one reversal, but (with some added variance) the results are the same as those from the paper. 

The plot produced at the end of the script replicates Figure 1 from the Wilson paper. 
"""

import numpy as np
import matplotlib.pyplot as plt 
import copy



def plot_extinction_errors(initial_sham, extinction_sham, initial_lesion, extinction_lesion):
    # find sum of arrays


    bar_width = 0.2
    opacity = 0.8
    print(np.sum(initial_sham))
    plt.bar(1, np.sum(initial_sham), bar_width,
    color='b',
    label='Pre-extinction, Sham Lesions')
     
    plt.bar(1.5, np.sum(initial_lesion), bar_width,
    color='r',
    label='Pre-extinction, OCF Lesions')
     
    plt.bar(2, np.sum(extinction_sham), bar_width,
    color='b',
    label='Post-Extinction, Sham Lesions')
     
    plt.bar(2.5, np.sum(extinction_lesion), bar_width,
    color='r',
    label='Post-Extinction, OCF Lesions')

    plt.ylabel('Number of Times Pressing Button Over {} Iterations'.format(500))
    plt.title('Extinction Performance')
    plt.tick_params(
        axis='x',          
        which='both',      
        bottom=False,      
        labelbottom=False)
    plt.legend()
     
    plt.tight_layout()
    plt.show()

def plot_postextinction(sham, sham_time, lesion, lesion_time):
    plt.plot(sham, sham_time, color='r', label='Sham Lesions')
    plt.plot(lesion, lesion_time, color='b', label='OCF Lesions')
    plt.tick_params( axis='x',          
        which='both',      
        bottom=False)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def select_action(q_learning_params, q_values, state):
    '''Select an action given the current state and using the Luce rule.'''
    # Compute the probability of choosing 'right'
    weighted_q_values = np.zeros(2)
    actions = ['press', 'no press']
    for action_index in range(2):
        q =  q_learning_params['beta']
        qv = q_values[str(state)][actions[action_index]]
        weighted_q_values[action_index] = q * qv
    prob_right = ((1.0 * np.exp(weighted_q_values[1])) / (1.0 * np.sum(np.exp(weighted_q_values))))
    # Choose an action based on the probability of choosing 'right' / 'left'
    return np.random.choice(['press', 'no press'], p=[1 - prob_right, prob_right])

def q_update(state_action_reward_dict, q_values, q_learning_params, state, action):
    '''Given the most recently taken action (and the resulting reward), update the stored q values.'''
    q_values[str(state)][action] += q_learning_params['alpha'] * (state_action_reward_dict[str(state)][action] - q_values[str(state)][action])
    return q_values 

def increment_press(state_action_reward_dict, state, action):
    if state == 1 and action == 'press':
        return 1
    else:
        return 0

def run_trial(state_action_reward_dict, q_learning_params, q_values, state, max_iters):
    '''Let the simulated agent take actions until they reach 90% correctness, and return how long it took them.'''
    state = 1
    previous_action = None
    num_pressed = [0] * max_iters

    for i in range(max_iters / 2):
        # Select an action
        action = select_action(q_learning_params, q_values, state)

        # Update the q-values 
        q_values = q_update(state_action_reward_dict, q_values, q_learning_params, state, action)
        
        # increment our accumulator for number pressed if pressed
        if action == 'press':
            num_pressed[i] = 1
    return num_pressed

def initialize_q_values(state_action_reward_dict):
    # Initialize q_values 
    if len(state_action_reward_dict) == 2:
        q_values = {'1': {'press': 0, 'no press': 0}, 
                    '2': {'press': 0, 'no press': 0}}
    else:
        q_values = {'1': {'press': 0, 'no press': 0}}
    return q_values

def extinction_learning(state_action_reward_dict, q_learning_params, q_values):
    '''Compute how many errors our simulated agent makes until they reach 90% accuracy in a reversal learning task.'''
    # Start in state 1
    state = 1
    # Take actions until 90% correct 
    presses_preextinction = run_trial(state_action_reward_dict, q_learning_params, q_values, state, 1000)  

    # Change state if sham lesioned
    if len(state_action_reward_dict) == 2:
        state = 2

    # Otherwise, remove reward for pressing lever
    else:
        state_action_reward_dict['1']['press'] = 0

    # runs trial post-extinction
    presses_extinction  = run_trial(state_action_reward_dict, q_learning_params, q_values, state, 1000)
    
    return presses_preextinction, presses_extinction

def reintroduction(state_action_reward_dict, q_learning_params, num_trials, q_values):
    pressed_array = [0] * num_trials
    time_values_array = [0] * num_trials


    initial_q = q_values.copy()

    for i in range(num_trials):

        pressed = 0

        # now, re-introduction of reward
        # assuming state selected with probability proportional to the time since it last occurred
        time_elapsed = np.random.randint(1, 100)

        # chooses initial state randomly
        initial_state = np.random.choice(['press', 'no press'], p=[1 - 1 / time_elapsed, 1 / time_elapsed])

        pressed = run_trial(state_action_reward_dict, q_learning_params, q_values, 1, 500)

        q_values = initial_q.copy()

        pressed_array[i] = pressed[-1]
        time_values_array[i] = time_elapsed
    return pressed_array, time_elapsed


def run_experiment():
    # Define q-learning parameters 
    q_learning_params = {
                        'alpha': 0.03, 
                        'beta': 3, 
                        'reward': 1
                        }

    # Set-up for each experimental condition
    sham_lesioned = {'1': {'press': q_learning_params['reward'], 'no press': 0}, 
                     '2': {'press': 0, 'no_press': 0}}
    OFC_lesioned = {'1': {'press': q_learning_params['reward'], 'no press': 0}}

    q_values_sham = initialize_q_values(sham_lesioned)
    q_values_OFC = initialize_q_values(OFC_lesioned)

    # Run experiment with sham lesioned agent
    initial_sham, extinction_sham = extinction_learning(sham_lesioned, q_learning_params, q_values_sham)

    # runs trial with re-instatement
    pressed_reintroduction_sham, time_elapsed_array_sham = reintroduction(sham_lesioned, q_learning_params, 500, q_values_sham)

    # runs extinction with OFC lesioned agent
    initial_lesion, extinction_lesion = extinction_learning(OFC_lesioned, q_learning_params, q_values_OFC)

    # runs trial with re-instatement
    pressed_reintroduction_sham, time_elapsed_array_sham = reintroduction(sham_lesioned, q_learning_params, 500, q_values_sham)

    # runs trial with re-instatement
    pressed_reintroduction_sham, time_elapsed_array_sham = reintroduction(OFC_lesioned, q_learning_params, 500, q_values_OFC)

    # Plot results for comparison with paper plots 
    plot_extinction_errors(initial_sham, extinction_sham, initial_lesion, extinction_lesion)
    plot_postextinction(pressed_reintroduction_sham, time_elapsed_array_sham, pressed_reintroduction_lesion, time_elapsed_array_lesion)

run_experiment()