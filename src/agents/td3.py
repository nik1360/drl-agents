import torch

from neural_networks.nn_utils import NetworkParams, soft_update, save_checkpoint, load_checkpoint
from neural_networks.critic import CriticTD3
from neural_networks.actor import Actor

from noises.exploration_noises import ParamNoise
from memory.replay_buffers import PrioritizedReplayBuffer


class TD3Agent:
    def __init__(self, params):
        self.gamma = params.gamma
        self.tau = params.tau
        self.batch_size = params.batch_size
        self.is_training = params.train
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.n_obs = params.n_obs
        self.n_actions = params.n_actions
        self.action_ub = torch.tensor(params.action_ub, device=self.device)
        self.action_lb = torch.tensor(params.action_lb, device=self.device)

        self.actor_dims=params.actor_dims
        self.critic_dims = params.critic_dims
        self.apply_input_norm = params.apply_input_norm
        self.apply_layer_norm = params.apply_layer_norm
        self.critic_lr = params.critic_lr
        self.actor_lr = params.actor_lr

        self.exploration_noise = params.exploration_noise
        self.training_noise = params.training_noise
        self.policy_delay = params.policy_delay
        self.replay_buffer = params.replay_buffer
        self.learn_it_count = 0 # learning  iteration counter

        self._init_networks()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr) # Set the optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr) # Set the optimizer
    
    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float, device=self.actor.device)
        
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor.forward(observation)
        self.actor.train()

        if self.is_training:
            mu_perturbed = self.exploration_noise.perturbate_action(action=mu, device=self.device, 
                **dict(observation=observation) if isinstance(self.exploration_noise, ParamNoise) else {})
            torch.clamp(input=mu_perturbed, min=self.action_lb, max=self.action_ub)
            return mu_perturbed.cpu().detach().numpy()
        else:
            return mu.cpu().detach().numpy()  
    
    def learn(self):
        if self.replay_buffer.mem_cntr < self.batch_size: 
            return

        # Sample experience from replay buffer 
        exp_batch = self.replay_buffer.sample_experience_batch(self.batch_size)
        exp_batch.to_tensors(self.device) # Convert the NumPy arrays into tensors
        exp_batch.rewards = exp_batch.rewards.view(self.batch_size, 1) # reshape from torch.Size([batch_size]) to torch.Size([batch_size, 1])
        exp_batch.not_dones = exp_batch.not_dones.view(self.batch_size, 1) # reshape from torch.Size([batch_size]) to torch.Size([batch_size, 1])

        self._optimize_critics(exp_batch=exp_batch)
        if (self.learn_it_count % self.policy_delay == 0) and (self.learn_it_count > 0):
            self._optimize_actor(exp_batch.states)

            soft_update(target_net=self.target_actor, origin_net=self.actor, tau=self.tau)
            soft_update(target_net=self.target_critic, origin_net=self.critic, tau=self.tau)
            
        self.learn_it_count += 1

    def remember(self, state, action, reward, new_state, done):
        self.replay_buffer.store_experience(state, action, reward, new_state, done)
    
    def save_models(self, checkpoint_dir, verbose):
        print("=====SAVING CHECKPOINTS====")
        save_checkpoint(checkpoint_dir, self.actor, verbose=verbose)
        save_checkpoint(checkpoint_dir, self.critic, verbose=verbose)
        save_checkpoint(checkpoint_dir, self.target_actor, verbose=verbose)
        save_checkpoint(checkpoint_dir, self.target_critic, verbose=verbose)

    def load_models(self, checkpoint_dir, verbose):
        print("=====LOADING CHECKPOINTS====")
        load_checkpoint(checkpoint_dir, self.actor, verbose=verbose)
        load_checkpoint(checkpoint_dir, self.critic, verbose=verbose)
        load_checkpoint(checkpoint_dir, self.target_actor, verbose=verbose)
        load_checkpoint(checkpoint_dir, self.target_critic, verbose=verbose)

    def _optimize_critics(self, exp_batch):
        
        mu_prime = self.target_actor.forward(exp_batch.new_states) 
        # Apply training noise 
        mu_prime = self.training_noise.perturbate_action(action=mu_prime, device=self.device)
        mu_prime = torch.clamp(mu_prime, self.action_lb, self.action_ub)

        Q_prime_1, Q_prime_2 = self.target_critic.forward(exp_batch.new_states, mu_prime) 
        Q_prime_min = torch.min(Q_prime_1, Q_prime_2)
        # y = torch.add(exp_batch.rewards, torch.mul(torch.mul(self.gamma, Q_prime_min), exp_batch.not_dones))
        y = exp_batch.rewards + self.gamma * Q_prime_min * exp_batch.not_dones

        Q1, Q2 = self.critic.forward(exp_batch.states, exp_batch.actions)
        
        td_error_1 = y - Q1
        td_error_2 = y - Q2

        loss1 = self._compute_weighted_mse(td_error = td_error_1, weights=exp_batch.weights)
        loss2 = self._compute_weighted_mse(td_error = td_error_2, weights=exp_batch.weights)
        
        critic_loss = loss1 + loss2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer): 
            self.replay_buffer.update_priorities(td_error_1.cpu().detach().numpy(), exp_batch.indeces)
        
    def _optimize_actor(self, states):
        mu = self.actor.forward(states)
        Q1, _ = self.critic.forward(states, mu) # loss = -Q
        actor_loss = torch.mean(-Q1) # perform the mean accross the minibatch
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward() # Backpropagate the loss 
        self.actor_optimizer.step() # Perform optimization step

    def _compute_weighted_mse(self, td_error, weights=None):
        if weights is None:
            weights = torch.ones_like(td_error, device=self.device)

        loss = torch.mul(1./self.batch_size, torch.sum( torch.mul(weights, torch.pow(td_error, 2))))
        return loss

    def _init_networks(self):
        critic_params = NetworkParams(n_inputs=self.n_obs, n_outputs=1, n_actions=self.n_actions,
            layer1_dims=self.critic_dims[0], layer2_dims=self.critic_dims[1], apply_input_norm=self.apply_input_norm,
            apply_layer_norm=self.apply_layer_norm, learning_rate=self.critic_lr)

        actor_params = NetworkParams(n_inputs=self.n_obs, n_outputs=self.n_actions, n_actions=self.n_actions,
            layer1_dims=self.actor_dims[0], layer2_dims=self.actor_dims[1], apply_input_norm=self.apply_input_norm,
            apply_layer_norm=self.apply_layer_norm, learning_rate=self.actor_lr, action_ub=self.action_ub)
        
        # Create the Networks  
        self.actor = Actor(params=actor_params, name="Actor", device=self.device)
        self.target_actor = Actor(params=actor_params, name="TargetActor", device=self.device)
        self.critic = CriticTD3(params=critic_params, name="Critic1", device=self.device)
        self.target_critic = CriticTD3(params=critic_params, name="TargetCritic1", device=self.device)
        

        # Initialize target networks parameters using hard copy
        soft_update(target_net=self.target_actor, origin_net=self.actor, tau=1)
        soft_update(target_net=self.target_critic, origin_net=self.critic, tau=1)
        