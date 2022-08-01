import numpy as np
import torch
import torch.nn as nn

from .nn_utils import init_layer_weights

class Actor(nn.Module):
    def __init__(self, params, name, device):
        super(Actor, self).__init__()

        self.input_dims = params.input_dims
        self.layer1_dims = params.layer1_dims
        self.layer2_dims = params.layer2_dims
        self.output_dims = params.output_dims 
        self.apply_input_norm = params.apply_input_norm  # Determine if BatchNorm is applied on inputs
        self.apply_layer_norm = params.apply_layer_norm  # Determine if LayerNorm is applied
        self.action_ub = params.action_ub

        self.name = name
        self.device = device

        self._build_network()
        self._init_network()

        self.to(self.device) # Load the Network on the device

    def _build_network(self):
        
        if self.apply_input_norm: self.bn_inputs = nn.BatchNorm1d(self.input_dims)
        self.layer1 = nn.Linear(self.input_dims, self.layer1_dims)
        if self.apply_layer_norm: self.layer1_norm = nn.LayerNorm(self.layer1_dims)
        self.layer2 = nn.Linear(self.layer1_dims, self.layer2_dims)
        if self.apply_layer_norm: self.layer2_norm = nn.LayerNorm(self.layer2_dims)
        self.mu = nn.Linear(self.layer2_dims, self.output_dims)

        self.relu = torch.nn.functional.relu
        self.tanh = torch.tanh

    def _init_network(self):
        init_layer_weights(self.layer1, 1./np.sqrt(self.layer1.weight.data.size()[0]))
        init_layer_weights(self.layer2, 1./np.sqrt(self.layer2.weight.data.size()[0]))
        init_layer_weights(self.mu, 0.003)

        
    def forward(self, observation):
        out = observation
        
        if self.apply_input_norm: self.bn_inputs(out) # Normalize Input

        out = self.layer1(out) # First layer
        if self.apply_layer_norm: out = self.layer1_norm(out)
        out = self.relu(out)
        
        out = self.layer2(out) # Second Layer
        if self.apply_layer_norm: out = self.layer2_norm(out)
        out = self.relu(out)
        
        out = self.mu(out) # Final Layer
        out = self.tanh(out) # Output between -1 and 1 (-100% and +100%)

        # Scale network output according to the action bounds
        out = torch.mul(out, self.action_ub) 
        return out