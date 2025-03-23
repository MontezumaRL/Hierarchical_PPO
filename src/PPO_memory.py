import numpy as np

class PPOMemory:

    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):

        print("===========================================")
        print("entering Generate batches")

        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # Ensure states are NumPy arrays and reshaped correctly for CNNs
        states_array = np.array(self.states, dtype=np.float32).reshape(n_states, 1, 84, 84)
        
        print("===========================================") 
        return states_array, np.array(self.actions), np.array(self.probs), \
           np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches


    def store_memory(self, state, action, probs, vals, reward, done):

        self.states.append(np.array(state, dtype=np.float32))  
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)


    def clear_memory(self):

        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

