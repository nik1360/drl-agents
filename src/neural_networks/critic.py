import numpy as np
import torch
import torch.nn as nn

from .nn_utils import init_layer_weights

class Critic(nn.Module):
    def __init__(self, params, name, device):
        super(Critic, self).__init__()
        
        self.input_dims = params.input_dims
        self.layer1_dims = params.layer1_dims
        self.layer2_dims = params.layer2_dims
        self.output_dims = params.output_dims
        self.n_actions = params.n_actions
        self.apply_input_norm = params.apply_input_norm  # Determine if BatchNorm is applied on inputs
        self.apply_layer_norm = params.apply_layer_norm  # Determine if LayerNorm is applied
        
        self.name = name
        self.device = device

        self._build_network()
        self._init_network()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.learning_rate) # Set the optimizer
        self.to(self.device) # Load the Network on the device

    def _build_network(self):
        if self.apply_input_norm: self.bn_inputs = nn.BatchNorm1d(self.input_dims)
        self.layer1 = nn.Linear(self.input_dims, self.layer1_dims)
        if self.apply_layer_norm: self.layer1_norm = nn.LayerNorm(self.layer1_dims)
        self.layer2 = nn.Linear(self.layer1_dims + self.n_actions, self.layer2_dims)
        if self.apply_layer_norm: self.layer2_norm = nn.LayerNorm(self.layer2_dims)
        self.q = nn.Linear(self.layer2_dims, self.output_dims)

        self.relu = torch.nn.functional.relu
    
    def _init_network(self):
        init_layer_weights(self.layer1, 1./np.sqrt(self.layer1.weight.data.size()[0]))
        init_layer_weights(self.layer2, 1./np.sqrt(self.layer2.weight.data.size()[0]))
        init_layer_weights(self.q, 0.003)

    def forward(self, observation, action):
        out = observation

        if self.apply_input_norm: self.bn_inputs(out) # Normalize Input

        out = self.layer1(observation) # First layer
        if self.apply_layer_norm: out = self.layer1_norm(out)
        out = self.relu(out)
        out = torch.cat([out, action], dim=1) # As detailed in the DDPG paper, action enters in the second layer
        out = self.layer2(out) # Second Layer
        if self.apply_layer_norm: out = self.layer2_norm(out)
        out = self.relu(out)

        out = self.q(out) # Final Layer
        return out

class CriticTD3(nn.Module):
    def __init__(self, params, name, device):
        super(CriticTD3, self).__init__()
        
        self.input_dims = params.input_dims
        self.layer1_dims = params.layer1_dims
        self.layer2_dims = params.layer2_dims
        self.output_dims = params.output_dims
        self.n_actions = params.n_actions
        self.apply_input_norm = params.apply_input_norm  # Determine if BatchNorm is applied on inputs
        self.apply_layer_norm = params.apply_layer_norm  # Determine if LayerNorm is applied
        
        self.name = name
        self.device = device

        self._build_network()
        self._init_network()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params.learning_rate) # Set the optimizer
        self.to(self.device) # Load the Network on the device

    def _build_network(self):
        # Build Q1 architecture
        if self.apply_input_norm: self.bn_inputs = nn.BatchNorm1d(self.input_dims)
        self.critic1_layer1 = nn.Linear(self.input_dims, self.layer1_dims)
        if self.apply_layer_norm: self.layer1_norm = nn.LayerNorm(self.layer1_dims)
        self.critic1_layer2 = nn.Linear(self.layer1_dims + self.n_actions, self.layer2_dims)
        if self.apply_layer_norm: self.layer2_norm = nn.LayerNorm(self.layer2_dims)
        self.critic1_q = nn.Linear(self.layer2_dims, self.output_dims)

        # Build Q2 architecture
        if self.apply_input_norm: self.bn_inputs = nn.BatchNorm1d(self.input_dims)
        self.critic2_layer1 = nn.Linear(self.input_dims, self.layer1_dims)
        if self.apply_layer_norm: self.layer1_norm = nn.LayerNorm(self.layer1_dims)
        self.critic2_layer2 = nn.Linear(self.layer1_dims + self.n_actions, self.layer2_dims)
        if self.apply_layer_norm: self.layer2_norm = nn.LayerNorm(self.layer2_dims)
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
        q1 = self._fwd_architecture(observation, action, self.critic1_layer1, self.critic1_layer2, self.critic1_q)
        q2 = self._fwd_architecture(observation, action, self.critic2_layer1, self.critic2_layer2, self.critic2_q)

        return q1, q2
    
    def _fwd_architecture(self, obs, action, layer1, layer2, layer3):
        out = obs

        if self.apply_input_norm: self.bn_inputs(out) # Normalize Input
        out = layer1(obs) # First layer
        if self.apply_layer_norm: out = self.layer1_norm(out)
        out = self.relu(out)
        out = torch.cat([out, action], dim=1) # As detailed in the DDPG paper, action enters in the second layer
        out = layer2(out) # Second Layer
        if self.apply_layer_norm: out = self.layer2_norm(out)
        out = self.relu(out)
        out = layer3(out) # Final Layer
        return out