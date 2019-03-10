# Neuro 1401
# Spring 2019
# Group 6: Project 2
import numpy as np
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, rewards):
        self.rewards = rewards
        self.Q = np.zeros_like(rewards).astype(float)
        self.state = None
        self.action = None

    def step(self, state, alpha=0.03, beta=3):
        actions = np.arange(self.rewards.shape[1])
        weights = np.exp(beta * self.Q[state])
        action = np.random.choice(actions, p=(weights / weights.sum()))
        self.Q[state, action] += alpha * (self.rewards[state, action] - self.Q[state, action])
        return action

    def run(self, *args, **kwargs):
        raise NotImplementedError()


class RepeatedAlternation(Simulation):
    def run(self, problems, actions):
        preference = np.random.choice(actions)
        goals, steps, errors = np.zeros((3, problems)).astype(int)

        # goals alternate every problem, starting with opposite of preference
        goals[1 - preference::2] = 1

        for i, goal in enumerate(goals):
            state = goal % self.rewards.shape[0]
            correct = np.zeros(12)

            self.rewards[state] = 0
            self.rewards[state, goal] = 1

            # Stop after 11 / 12 past trials correct including most recent 8
            while np.sum(correct[-8:]) < 8 and np.sum(correct) < 11:
                action = self.step(state)

                # Update most recent correct, correct action if it was best of all actions
                correct = np.roll(correct, 1)
                correct[0] = self.rewards[state, action] == self.rewards[state].max()
                steps[i] += 1
                errors[i] += 1 - correct[0]
        return errors / steps


trials = 12
sham = np.array([[0, 1], [1, 0]])
OFC = np.array([[0, 1]])

LEFT, RIGHT = [0, 1]
actions = [LEFT, RIGHT]

mean_error_sham = RepeatedAlternation(sham).run(trials, actions)
mean_error_ofc = RepeatedAlternation(OFC).run(trials, actions)

xs = np.arange(trials)

plt.plot(xs, mean_error_sham, label='Sham Lesions')
plt.plot(xs, mean_error_ofc, label='OFC Lesions')
plt.title('Action Error Over Repeated Reversals')
plt.ylabel('Average Action Error')
plt.xlabel('Trial Number')
plt.legend()
plt.tight_layout()
plt.show()
