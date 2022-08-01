from agents.ddpg import DDPGAgent

from utils import parse_arguments, exploration_noise_from_args, replay_buffer_from_args
import gym
import numpy as np

if __name__ == "__main__":
    args = parse_arguments()
    env = gym.make(args.env_name)

    exploration_noise = exploration_noise_from_args(args=args, n_actions=env.action_space.shape[0], n_obs=env.observation_space.shape[0], action_ub=env.action_space.high)
    replay_buffer = replay_buffer_from_args(args=args, n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0])

    agent = DDPGAgent(train_agent=args.train_agent, actor_lr=args.actor_lr, critic_lr=args.critic_lr, 
        actor_dims=[args.actor_l1_dim, args.actor_l2_dim], critic_dims=[args.critic_l1_dim, args.critic_l2_dim],  
        batch_size=args.batch_size, gamma=args.discount_factor, tau=args.soft_update_factor, 
        n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], action_ub=env.action_space.high,
        action_lb=env.action_space.low, exploration_noise=exploration_noise, replay_buffer=replay_buffer)

    score_history = []

    for i in range(args.train_episodes):
        
        agent.exploration_noise.reset(**( dict(actor_net=agent.actor, replay_buffer=agent.replay_buffer, 
            batch_size=agent.batch_size) if args.noise_type=="PARAM" else {}))

        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state

        score_history.append(score)
        print('Episode: ', i, ' => Score %.2f' % score,
            '| Average 100 episodes: %.3f' % np.mean(score_history[-100:]))
    
    