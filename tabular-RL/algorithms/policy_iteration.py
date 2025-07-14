import numpy as np

class PolicyIteration:

    def __init__(self, states, actions, tpm, rewards, time_steps, discount=1.0):
        self.states = states
        self.actions = actions
        self.tpm = tpm
        self.rewards = rewards
        self.ts = time_steps
        self.discount = discount

        self.policy = np.zeros((self.ts, self.states), dtype=int)
        self.V = np.zeros((self.ts + 1, self.states))
    
    def policy_evaluation(self):
        for t in range(self.ts - 1, -1, -1): # iteration for t-1 to 0
            for s in range(self.states):
                a = self.policy[t, s]
                self.V[t, s] = self.rewards[s, a] +self.discount * np.sum(self.tpm[s, a, :] * self.V[t + 1, :])
    
    def policy_improvement(self):
        policy_stable = True
        for t in range(self.ts):
            for s in range(self.states):
                prev_act = self.policy[t, s]
                q_vals = np.zeros(self.actions)
                for a in range(self.actions):
                    q_vals[a] = self.rewards[s, a] + self.discount * np.sum(self.tpm[s, a, :] * self.V[t + 1, :])
                best_act = np.argmax(q_vals)
                self.policy[t, s] = best_act
                if prev_act != best_act:
                    policy_stable = False
        return policy_stable
    
    def run(self):
        while (True):
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy, self.V