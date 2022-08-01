from agents.ddpg import DDPGAgent

from utils.init_utils import parse_arguments, exploration_noise_from_args, replay_buffer_from_args
from utils.drl_utils import perform_training, perform_testing
from utils.print_plot_utils import print_run_parameters
import gym

if __name__ == "__main__":
    args = parse_arguments()
    env = gym.make(args.env_name )

    train_agent = args.train_agent=="True" 

    exploration_noise = exploration_noise_from_args(args=args, n_actions=env.action_space.shape[0], n_obs=env.observation_space.shape[0], action_ub=env.action_space.high)
    replay_buffer = replay_buffer_from_args(args=args, n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])

    agent = DDPGAgent(train_agent=args.train_agent, actor_lr=args.actor_lr, critic_lr=args.critic_lr, 
        actor_dims=[args.actor_l1_dim, args.actor_l2_dim], critic_dims=[args.critic_l1_dim, args.critic_l2_dim],  
        batch_size=args.batch_size, gamma=args.discount_factor, tau=args.soft_update_factor, 
        n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], action_ub=env.action_space.high,
        action_lb=env.action_space.low, exploration_noise=exploration_noise, replay_buffer=replay_buffer)

    print_run_parameters(args)
    print("Press a key to continue...")
    input()

    if train_agent:
        perform_training(agent=agent, train_episodes=args.train_episodes, noise_type=args.noise_type, env=env)
    else:
        perform_testing(agent=agent, test_episodes=args.test_episodes, env=env)    