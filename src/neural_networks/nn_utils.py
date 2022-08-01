import os
import numpy as np
import torch
import torch.nn as nn

def soft_update(target_net, origin_net, tau):
    for target_param, origin_param in zip(target_net.parameters(), origin_net.parameters()):
        target_param.data.copy_(tau*origin_param.data + (1.0-tau)*target_param.data)

def init_layer_weights(layer, bound):
    nn.init.uniform_(layer.weight.data, -bound, bound)
    nn.init.uniform_(layer.bias.data, -bound, bound)

def save_checkpoint(checkpoint_dir, network, verbose):
        if verbose: print(network.name + ' Saving checkpoint ...')
        checkpoint_file = os.path.join(checkpoint_dir, network.name+'_ddpg')
        torch.save(network.state_dict(), checkpoint_file)

def load_checkpoint(checkpoint_dir, network, verbose):
        if verbose: print(network.name + ' Loading checkpoint ...')
        checkpoint_file = os.path.join(checkpoint_dir, network.name+'_ddpg')
        network.load_state_dict(torch.load(checkpoint_file))

class NetworkParams:
    def __init__(self, n_inputs, layer1_dims, layer2_dims, n_outputs, n_actions, 
        action_ub=None):
        self.input_dims = n_inputs
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.output_dims = n_outputs
        self.n_actions = n_actions
        self.action_ub = action_ub

