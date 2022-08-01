from abc import ABC, abstractmethod
from random import sample
import numpy as np

from memory.experience_batch import ExperienceBatch



class ReplayBuffer(ABC):
    def __init__(self, size, n_actions, n_obs):
        super().__init__()
        self.max_mem_size = size
        self.mem_cntr = 0 # Count how many elements have been inserted in the buffer
        self.state_memory = np.zeros((self.max_mem_size, n_obs)) 
        self.new_state_memory = np.zeros((self.max_mem_size, n_obs)) 
        self.action_memory = np.zeros((self.max_mem_size, n_actions)) 
        self.reward_memory = np.zeros(self.max_mem_size)
        self.not_done_memory = np.zeros(self.max_mem_size, dtype=np.float32) # Flag hich indicates if the state is terminal

    @abstractmethod
    def store_experience(self):
        pass

    @abstractmethod
    def sample_experience_batch(self):
        pass

class UniformReplayBuffer(ReplayBuffer):
    def __init__(self, size, n_actions, n_obs):
        super().__init__(size, n_actions, n_obs)
    
    def store_experience(self, state, action, reward, new_state, done):
        # Compute the index of the memory in which experience is saved.
        # When the memory is full, the oldest experience is discarded.
        index = self.mem_cntr % self.max_mem_size 
        
        # Store the experience
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.not_done_memory[index] = 1 - done 

        # Increase the counter of Experience stored 
        self.mem_cntr += 1

    def sample_experience_batch(self, batch_size):
        
        # Check the number of experiences inserted in the buffer
        exp_in_mem = min(self.mem_cntr, self.max_mem_size)

        # Select batch_size random indeces from the buffer
        batch_indeces = np.random.choice(range(0, exp_in_mem), batch_size)

        states = self.state_memory[batch_indeces]
        actions = self.action_memory[batch_indeces]
        rewards = self.reward_memory[batch_indeces]
        new_states = self.new_state_memory[batch_indeces]
        not_dones = self.not_done_memory[batch_indeces]

        return ExperienceBatch(states, actions, rewards, new_states, not_dones, batch_indeces)
        

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, n_actions, n_obs, alpha=0.6, beta=0.4, beta_increment=None):
        super().__init__(size, n_actions, n_obs)
        self.priorities = np.zeros(self.max_mem_size)
        self.scaled_priorities = np.zeros(self.max_mem_size)
        self.sum_scaled_priorities = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
    
    def store_experience(self, state, action, reward, new_state, done):
        # Compute the index of the memory in which experience is saved.
        # When the memory is full, the oldest experience is discarded.
        index = self.mem_cntr % self.max_mem_size 
        
        # Store the experience
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.not_done_memory[index] = 1 - done 

        if self.mem_cntr >= self.max_mem_size: 
            self.priorities[index] = np.max(self.priorities)
        else: 
            self.priorities[index] = np.max(self.priorities[0:index]) + 0.01 if self.mem_cntr > 0 else 1

        self.scaled_priorities[index] = self.priorities[index]**self.alpha
        self.sum_scaled_priorities += self.scaled_priorities[index]

        # Increase the counter of Experience stored 
        self.mem_cntr += 1


    def sample_experience_batch(self, batch_size):
        # Check the number of experiences inserted in the buffer
        exp_in_mem = min(self.mem_cntr, self.max_mem_size)
        
        
        sample_prob = self.scaled_priorities[0:exp_in_mem]/self.sum_scaled_priorities
        
        # Select batch_size random indeces from the buffer accrding to the computed sample probabilities
        batch_indeces = np.random.choice(range(0, exp_in_mem), size=batch_size, p=sample_prob)

        weights = (exp_in_mem * sample_prob[batch_indeces])**(-self.beta) 
        weights = weights / np.max(weights)

        states = self.state_memory[batch_indeces]
        actions = self.action_memory[batch_indeces]
        rewards = self.reward_memory[batch_indeces]
        new_states = self.new_state_memory[batch_indeces]
        not_dones = self.not_done_memory[batch_indeces]

        return ExperienceBatch(states, actions, rewards, new_states, not_dones, batch_indeces, weights)

    def update_priorities(self, td_error, batch_indeces):
        # self.sum_scaled_priorities -= np.sum(self.scaled_priorities[batch_indeces])

        self.priorities[batch_indeces] = np.reshape(np.fabs(td_error), (len(batch_indeces, )))

        self.scaled_priorities[batch_indeces] = self.priorities[batch_indeces]**self.alpha
        self.sum_scaled_priorities = np.sum(self.scaled_priorities)

        self.beta += self.beta_increment


class HindsightReplayBuffer(ReplayBuffer):
    def __init__(self, size, n_actions, n_obs):
        super().__init__(size, n_actions, n_obs)
    
    def store_experience(self):
        pass

    def sample_experience_batch(self):
        pass