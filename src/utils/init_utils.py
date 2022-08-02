import argparse
import numpy as np
from noises.exploration_noises import OUNoise, ParamNoise, RandNoise
from memory.replay_buffers import UniformReplayBuffer, HindsightReplayBuffer, PrioritizedReplayBuffer
from neural_networks.nn_utils import NetworkParams
from neural_networks.nn_ddpg import Actor
import torch


def parse_arguments(td3=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_save_dir", type=str, default="../data/saved_checkpoints", help="Directory in which checkpoints are saved")
    parser.add_argument("--checkpoint_load_dir", type=str, default="../data/to_load_checkpoints", help="Directory in which checkpoints are loaded_from")
    parse_arguments_ddpg(parser)
    if td3:
        parse_arguments_td3(parser=parser)
    
    args = parser.parse_args()
    return args

def parse_arguments_ddpg(parser):
    parser.add_argument("--actor_lr", type=float, default=0.0001, help="Actor Network Learning Rate")
    parser.add_argument("--critic_lr", type=float, default=0.001, help="Critic Network Learning Rate")
    parser.add_argument("--actor_l1_dim", type=int,  default=400, help="Dimension of the first layer of the Actor ")
    parser.add_argument("--actor_l2_dim", type=int,  default=300, help="Dimension of the second layer of the Actor ")
    parser.add_argument("--critic_l1_dim", type=int, default=400, help="Dimension of the first layer of the Critic ")
    parser.add_argument("--critic_l2_dim", type=int, default=300, help="Dimension of the second layer of the Critic")
    parser.add_argument("--noise_type", type=str, default='OU', choices=['OU', 'PARAM', 'RAND'], help="Type of exploration noise")
    parser.add_argument("--param_noise_desired_distance", type=float, default=0.2, help="Desired distance for the parameter space exploration noise")
    parser.add_argument("--param_noise_decay_factor", type=float, default=1.1, help="Decay factor for the parameter space exploration noise")
    parser.add_argument("--param_noise_initial_scale", type=float, default=0.1, help="Initial scale parameter space exploration noise")
    parser.add_argument("--ou_noise_sigma", type=float, default=0.15, help="Sigma value for the Ornstein–Uhlenbeck action space noise")
    parser.add_argument("--ou_noise_theta", type=float, default=0.2, help="Theta value for the Ornstein–Uhlenbeck action space noise")
    parser.add_argument("--rand_noise_std", type=float, default=0.15, help="standard deviation for the normal random action space noise")
    parser.add_argument("--experience_replay_type", type=str, default='UNIFORM', choices=['UNIFORM', 'PRIORITIZED', 'HINDSIGHT'], help="Type of Experience Replay")
    parser.add_argument("--replay_buffer_size", type=int, default=100000, help="Experience replay buffer size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--soft_update_factor", type=float, default=0.001, help="Factor for the soft update of the target networks")
    parser.add_argument("--discount_factor", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_agent", type=str, default="True", help="Train the agent")
    parser.add_argument("--train_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--test_episodes", type=int, default=10, help="Number of testing episodes")
    parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v2", help="Name of the OpenAI Gym environment")

        
def parse_arguments_td3(parser):
    parser.add_argument("--policy_delay", type=int, default=2, help="Frequency pof actor and targets update")
    parser.add_argument("--train_noise_std", type=float, default=0.1, help="Standard deviation of the training noise")
    parser.add_argument("--train_noise_clip", type=float, default=0.5, help="Maximum value for the training noise")
 
def exploration_noise_from_args(args, n_actions, n_obs, action_ub):
    if args.noise_type == "OU":
        noise = OUNoise(mu=np.zeros(n_actions), sigma=args.ou_noise_sigma, theta=args.ou_noise_theta)
    elif args.noise_type == "PARAM":
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        action_ub_tens = torch.tensor(action_ub, device=device)
        perturbed_actor_params = NetworkParams(n_inputs=n_obs, n_outputs=n_actions, n_actions=n_actions,
            layer1_dims=args.actor_l1_dim, layer2_dims=args.actor_l2_dim, action_ub=action_ub_tens)

        perturbed_actor = Actor(params=perturbed_actor_params, name="PerturbedActor", device=device)

        noise = ParamNoise(desired_distance=args.param_noise_desired_distance, decay_factor=args.param_noise_decay_factor,
            initial_scale=args.param_noise_initial_scale, perturbed_actor=perturbed_actor)
        
    elif args.noise_type == "RAND":
        noise = RandNoise(std=args.rand_noise_std)

    return noise

def training_noise_from_args(args):
    noise = RandNoise(std=args.train_noise_std, clip=args.train_noise_clip)
    return noise

def replay_buffer_from_args(args, n_obs, n_actions):
    if args.experience_replay_type == "UNIFORM":
        replay_buffer = UniformReplayBuffer(size=args.replay_buffer_size, n_obs=n_obs, n_actions=n_actions)
    elif args.experience_replay_type ==  "PRIORITIZED":
        beta_increment = (1 - 0.4)/args.train_episodes
        replay_buffer = PrioritizedReplayBuffer(size=args.replay_buffer_size, n_obs=n_obs, n_actions=n_actions, beta_increment = beta_increment)
    elif args.experience_replay_type ==  "HINDSIGHT":
        replay_buffer = HindsightReplayBuffer(size=args.replay_buffer_size, n_obs=n_obs, n_actions=n_actions)
    return replay_buffer