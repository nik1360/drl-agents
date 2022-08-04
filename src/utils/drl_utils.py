import numpy as np
import os

def perform_training(agent, train_episodes, noise_type, env, checkpoint_dir, verbose=True, mean_window=100, 
    save_frequency=50):
    score_history = []

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    for i in range(train_episodes):
        agent.exploration_noise.reset(**( dict(actor_net=agent.actor, replay_buffer=agent.replay_buffer, 
            batch_size=agent.batch_size) if noise_type=="PARAM" else {}))

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
        if (i % save_frequency == 0):
            agent.save_models(checkpoint_dir=checkpoint_dir, verbose=True)
        
        if verbose:
            print('Episode: ', i, ' => Score %.2f' % score,
                '| Average 100 episodes: %.3f' % np.mean(score_history[-mean_window:]))
    return score_history


def perform_testing(agent, test_episodes,  checkpoint_dir, env, verbose=True):
    score_history = []

    agent.load_models(checkpoint_dir=checkpoint_dir, verbose=verbose)
    for i in range(test_episodes):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)

            score += reward
            obs = new_state

        score_history.append(score)
        if verbose:
            print('Episode: ', i, ' => Score %.2f' % score)
    
    return score_history