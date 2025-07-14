import numpy as np

class ValueIteration:
    
    def __init__(self, states, actions, tpm, rewards, time_steps, discount=1.0):
        self.states = states
        self.actions = actions
        self.tpm = tpm
        self.rewards = rewards
        self.ts = time_steps
        self.discount = discount

        self.V = np.zeros((self.ts + 1, self.states))
        self.policy = np.zeros((self.ts, self.states), dtype=int)

    def value_iteration(self):
        for t in range(self.ts - 1, - 1, -1): #iteration backwards from t-1 to 0
            for s in range(self.states):
                q_vals = np.zeros(self.actions)
                for a in range(self.actions):
                    q_vals[a] = self.rewards[s, a] + self.discount * np.sum(self.tpm[s, a, :]*self.V[t + 1, :])
                    self.V[t, s] = np.max(q_vals)
                    self.policy[t, s] = np.argmax(q_vals)
                    return self.policy, self.V
