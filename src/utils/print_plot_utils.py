from tabulate import tabulate

def print_run_parameters(args, td3=False):

    params_list = []
    params_list.append(['Environment', args.env_name])
    if (args.train_agent=='True'):
        params_list.append(['Train Agent', 1])
        params_list.append(['Training episodes', args.train_episodes])
    else:
        params_list.append(['Train Agent', 0])
        params_list.append(['Training episodes', args.test_episodes])
    params_list.append(['Actor learning rate', args.actor_lr])
    params_list.append(['Critic learning rate', args.critic_lr])
    params_list.append(['Actor layer 1 size', args.actor_l1_dim])
    params_list.append(['Actor layer 2 size', args.actor_l2_dim])
    params_list.append(['Critic layer 1 size', args.critic_l1_dim])
    params_list.append(['Critic layer 2 size', args.critic_l2_dim])
    params_list.append(['Exploration noise type', args.noise_type])
    if args.noise_type == 'OU':
        params_list.append(['Ornstein Uhlenbeck noise sigma', args.ou_noise_sigma])
        params_list.append(['Ornstein Uhlenbeck noise theta', args.ou_noise_theta])
    elif args.noise_type == 'PARAM':
        params_list.append(['Parameter space noise desired distance', args.param_noise_desired_distance])
        params_list.append(['Parameter space noise decay factor', args.param_noise_decay_factor])
        params_list.append(['Parameter space noise initial scale', args.param_noise_initial_scale])
    elif args.noise_type == 'RAND':
        params_list.append(['Random noise standard deviation', args.rand_noise_std])
    params_list.append(['Experience replay type', args.experience_replay_type])
    params_list.append(['Replay buffer size', args.replay_buffer_size])
    params_list.append(['Batch size', args.batch_size])
    params_list.append(['Soft update factor tau', args.soft_update_factor])
    params_list.append(['Discount factor gamma', args.discount_factor])
    if td3:
        params_list.append(['Policy delay', args.policy_delay])
        params_list.append(['Training noise standard deviation', args.train_noise_std])
        params_list.append(['Training noise clip', args.train_noise_clip])
    tab = tabulate(params_list, headers=['Parameter', 'Value'], tablefmt="grid")
    print(tab)