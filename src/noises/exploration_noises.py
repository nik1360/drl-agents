from abc import ABC, abstractmethod

import torch
import numpy as np

from neural_networks.actor import Actor
from neural_networks.nn_utils import soft_update
 
class ExplorationNoise(ABC):
    @abstractmethod
    def perturbate_action(self, action):
        pass
    @abstractmethod
    def reset(self):
        pass

'''Class which implements a Ornstein Uhlenbeck process to generate an action noise'''
class OUNoise(ExplorationNoise):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        super().__init__()
        self.sigma = sigma
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.reset()

    def perturbate_action(self, action, device):
        noise = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = noise
        perturbed_action = action + torch.tensor(noise, device=device)

        return perturbed_action
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ParamNoise(ExplorationNoise):
    def __init__(self, desired_distance, decay_factor, initial_scale, perturbed_actor):
        super().__init__()
        self.desired_distance = desired_distance
        self.scale = initial_scale
        self.decay_factor = decay_factor
        self.perturbed_actor = perturbed_actor

    
    def perturbate_action(self, action, device, observation):
        
        perturbed_action = self.perturbed_actor(observation)
        return perturbed_action
    
    def _perturbate_actor_net(self, actor):
        # Create a copy of the actor 
        soft_update(self.perturbed_actor, actor, tau=1) # Perform Hard Copy
        # self.perturbed_actor.load_state_dict(actor.state_dict().copy())
        # Perturbate layer 1
        self.perturbed_actor.layer1.weight.data += torch.randn_like(self.perturbed_actor.layer1.weight.data, device=self.perturbed_actor.device) * (self.scale)
        # Perturbate layer 2
        self.perturbed_actor.layer2.weight.data += torch.randn_like(self.perturbed_actor.layer2.weight.data, device=self.perturbed_actor.device) * (self.scale)
        # Perturbate layer 3
        self.perturbed_actor.mu.weight.data += torch.randn_like(self.perturbed_actor.mu.weight.data, device=self.perturbed_actor.device) * (self.scale)
    
    def _adapt_noise_scale(self, actor_net, replay_buffer, batch_size):
        if replay_buffer.mem_cntr < batch_size: 
            return
        # Sample a mini batch from the replay buffer 
        exp_batch = replay_buffer.sample_experience_batch(batch_size)
        exp_batch.to_tensors(self.perturbed_actor.device)

        actor_net.eval()
        self.perturbed_actor.eval()
        with torch.no_grad():
            mu = actor_net.forward(exp_batch.states)
            mu_perturbed = self.perturbed_actor(exp_batch.states)
        actor_net.train()
        self.perturbed_actor.train()
        # Compute the distance between the output of the Actor and Perturbed Actor
        diff = mu - mu_perturbed
        batch_mean = torch.mean(torch.pow(diff, 2), axis=0) # mean value accross minibatch
        act_mean = torch.mean(batch_mean) # mean value accross actions
        distance = torch.sqrt(act_mean).item()

        # Update the noise scale
        if distance < self.desired_distance:
            self.scale = self.scale * self.decay_factor
        if distance > self.desired_distance:
            self.scale = self.scale / self.decay_factor

    def reset(self, actor_net, replay_buffer, batch_size):
        self._perturbate_actor_net(actor_net)
        self._adapt_noise_scale(actor_net, replay_buffer, batch_size)
        

class RandNoise(ExplorationNoise):
    def __init__(self, std, clip=None):
        super().__init__()
        self.std = std
        self.clip = clip
    
    def perturbate_action(self, action, device):
        noise = torch.empty(action.shape, device=device).normal_(mean=0,std=self.std)
        if self.clip is not None:
            noise = torch.clamp(noise, -self.clip, self.clip)
        perturbed_action = action + noise
        return perturbed_action
    
    def reset(self):
        pass