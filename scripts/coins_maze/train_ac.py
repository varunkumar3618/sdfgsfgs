import sacred
import torch
import torch.nn as nn

from graphrl.sacred.config import add_params
from graphrl.agents.actor_critic_agent import ActorCriticAgentParams
from graphrl.modules.heads import CategoricalValueHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.coins_maze.coins_maze_v0 import make_coins_maze_env
from graphrl.environments.wrappers import RenderEnv


ex = sacred.Experiment('train_coins_maze_ac')


@ex.config
def arch_config():
    arch = {
        'trunk': ALL_TRUNK_CONFIG
    }


@ex.config
def config():
    env = {
        'artfile': 'assets/coins_maze_art/6x7/art_6x7_1.txt',
        'render': False
    }

    agent = {
        'device_name': 'cuda',
        'policy_entropy_weight': 0.01,
        'update_freq_counter': 'steps',
        'update_freq_steps': 5,
        'max_steps_per_test_episode': 100,
        'gamma': 0.99
    }

    arch = {
        'trunk': {
            'trunk_type': 'mlp',
            'hidden_sizes': [512]
        }
    }

    opt = {
        'kwargs': {
            'lr': 1e-4
        }
    }


add_params = ex.capture(add_params)


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = CategoricalValueHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


@ex.capture(prefix='env')
def build_env(artfile, render):
    env = make_coins_maze_env(artfile, encode_onehot=True)
    if render:
        env = RenderEnv(env)
    return env


@ex.automain
def main(_seed, _run):
    torch.manual_seed(_seed)

    env = build_env()
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent_params = ActorCriticAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')
    agent_params.sacred_run = _run
    agent_params.env = env
    agent_params.mode = 'train'

    policy_value_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net

    agent = agent_params.make_agent()
    agent.run()
