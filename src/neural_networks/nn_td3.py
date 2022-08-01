import numpy as np
import torch
import torch.nn as nn

from .nn_utils import init_layer_weights

# The Critic ntwork is composed by two different architectures with the same struture
class Critic(nn.Module):
    def __init__(self, params, name, device):
        super(Critic, self).__init__()
        
        self.input_dims = params.input_dims
        self.layer1_dims = params.layer1_dims
        self.layer2_dims = params.layer2_dims
        self.output_dims = params.output_dims
        self.n_actions = params.n_actions
        
        self.name = name
        self.device = device

        self._build_network()
        self._init_network()

        self.to(self.device) # Load the Network on the device

    def _build_network(self):
        self.critic1_layer1 = nn.Linear(self.input_dims + self.n_actions, self.layer1_dims)
        self.critic1_layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.critic1_q = nn.Linear(self.layer2_dims, self.output_dims)

        self.critic2_layer1 = nn.Linear(self.input_dims + self.n_actions, self.layer1_dims)
        self.critic2_layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.critic2_q = nn.Linear(self.layer2_dims, self.output_dims)

        self.relu = torch.nn.functional.relu
    
    def _init_network(self):
        init_layer_weights(self.critic1_layer1, 1./np.sqrt(self.critic1_layer1.weight.data.size()[0]))
        init_layer_weights(self.critic1_layer2, 1./np.sqrt(self.critic1_layer2.weight.data.size()[0]))
        init_layer_weights(self.critic1_q, 0.003)

        init_layer_weights(self.critic2_layer1, 1./np.sqrt(self.critic2_layer1.weight.data.size()[0]))
        init_layer_weights(self.critic2_layer2, 1./np.sqrt(self.critic2_layer2.weight.data.size()[0]))
        init_layer_weights(self.critic2_q, 0.003)

    def forward(self, observation, action):
        q1 = self.Q1(observation, action)
        q2 = self.Q2(observation, action)
        return q1, q2

    def Q1(self, obs, action):
        q1 = torch.cat([obs, action], dim=1)
        q1 = self.relu(self.critic1_layer1(q1)) # First layer
        q1 = self.relu(self.critic1_layer2(q1)) # Second Layer
        q1 = self.critic1_q(q1) # Output layer
        return q1

    def Q2(self, obs, action):
        q2 = torch.cat([obs, action], dim=1)
        q2 = self.relu(self.critic1_layer1(q2)) # First layer
        q2 = self.relu(self.critic1_layer2(q2)) # Second Layer
        q2 = self.critic1_q(q2) # Output layer
        return q2

class Actor(nn.Module):
    def __init__(self, params, name, device):
        super(Actor, self).__init__()

        self.input_dims = params.input_dims
        self.layer1_dims = params.layer1_dims
        self.layer2_dims = params.layer2_dims
        self.output_dims = params.output_dims 
        self.action_ub = params.action_ub

        self.name = name
        self.device = device

        self._build_network()
        self._init_network()

        self.to(self.device) # Load the Network on the device

    def _build_network(self):
        self.layer1 = nn.Linear(self.input_dims, self.layer1_dims)
        self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        self.mu = nn.Linear(self.layer2_dims, self.output_dims)

        self.relu = torch.nn.functional.relu
        self.tanh = torch.tanh

    def _init_network(self):
        init_layer_weights(self.layer1, 1./np.sqrt(self.layer1.weight.data.size()[0]))
        init_layer_weights(self.layer2, 1./np.sqrt(self.layer2.weight.data.size()[0]))
        init_layer_weights(self.mu, 0.003)
        
    def forward(self, observation):
        out = observation
        
        out = self.relu(self.layer1(out)) # First layer
        out = self.relu(self.layer2(out)) # Second Layer
        out = self.tanh(self.mu(out)) # Final Layer
                                      # Output between -1 and 1 (-100% and +100%)

        # Scale network output according to the action bounds
        out = torch.mul(out, self.action_ub) 
        return out