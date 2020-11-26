from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.save_index = 100
        
    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if  len(self.buffer) < batch_size:
            return self.buffer
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state):
        experience = (state, action, reward, new_state)
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.save_data()
            preserve_no = self.buffer_size // 10
            self.buffer = random.sample(self.buffer, preserve_no)
            self.buffer.append(experience)
            
    def count(self):
        return len(self.buffer)

    def erase(self):
        self.buffer = deque()

    def save_data(self):
        batch_states = np.asarray([e[0] for e in self.buffer]).reshape(len(self.buffer),-1)
        batch_actions = np.asarray([e[1] for e in self.buffer]).reshape(len(self.buffer),-1)
        batch_rewards = np.asarray([e[2] for e in self.buffer]).reshape(len(self.buffer),-1)
        batch_next_states = np.asarray([e[3] for e in self.buffer]).reshape(len(self.buffer),-1)   
        np.save('saved_data/n_batch_states_'+ str(self.save_index)+'.npy',batch_states)
        np.save('saved_data/n_batch_actions_'+ str(self.save_index)+'.npy',batch_actions)
        np.save('saved_data/n_batch_rewards_'+ str(self.save_index)+'.npy',batch_rewards)
        np.save('saved_data/n_batch_next_states_'+ str(self.save_index)+'.npy',batch_next_states)        
        self.save_index = self.save_index + 1
        
        
        
        
        