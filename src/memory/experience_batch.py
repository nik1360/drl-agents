
import torch

class ExperienceBatch:
    def __init__(self, states, actions, rewards, new_states, not_dones, indeces, weights=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.new_states = new_states
        self.not_dones = not_dones
        self.indeces = indeces
        self.weights = weights # Used to compensate the bias in PRioritized Experience replay
    
    def to_tensors(self, device):
        self.states = torch.tensor(self.states, dtype=torch.float, device=device)
        self.actions = torch.tensor(self.actions, dtype=torch.float, device=device)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float, device=device)
        self.new_states = torch.tensor(self.new_states, dtype=torch.float, device=device)
        self.not_dones = torch.tensor(self.not_dones, device=device)
        if self.weights is not None:
            self.weights = torch.tensor(self.weights, device=device) # Used to compensate the bias in Prioritized Experience replay