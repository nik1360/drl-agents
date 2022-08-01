class DDPGParams:
    def __init__(self, train, actor_lr, critic_lr, actor_dims, critic_dims, apply_input_norm, apply_layer_norm, batch_size,
        gamma, tau, n_obs, n_actions, action_ub,  action_lb,  noise, replay_buffer):
        self.train = train
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_dims=actor_dims
        self.critic_dims = critic_dims
        self.apply_input_norm = apply_input_norm
        self.apply_layer_norm = apply_layer_norm
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.action_ub = action_ub
        self.action_lb = action_lb
        self.exploration_noise = noise
        self.replay_buffer = replay_buffer


class TD3Params(DDPGParams):
    def __init__(self, train, actor_lr, critic_lr, actor_dims, critic_dims, apply_input_norm, apply_layer_norm, batch_size,
        gamma, tau, n_obs, n_actions, action_ub,  action_lb,  exploration_noise, training_noise, policy_delay, replay_buffer):
        
        super().__init__(train, actor_lr, critic_lr, actor_dims, critic_dims, apply_input_norm, apply_layer_norm, batch_size,
        gamma, tau, n_obs, n_actions, action_ub,  action_lb, exploration_noise, replay_buffer)
        
        self.training_noise = training_noise
        self.policy_delay = policy_delay
       