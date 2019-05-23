import gym
import sacred
import torch
import torch.nn as nn

from graphrl.agents.filters import RewardRescaleFilter
from graphrl.agents.policy_gradients_agent import PolicyGradientsAgentParams
from graphrl.sacred.config import add_params
from graphrl.modules.heads import CategoricalHead
from graphrl.modules.trunks import build_trunk


ex = sacred.Experiment('train_cartpole_pg')


@ex.config
def config():
    agent = {
        'update_freq_counter': 'episodes',
        'update_freq_episodes': 5,
        'returns_normalizer': 'timestep',
        'device_name': 'cuda'
    }

    arch = {
        'trunk': {
            'trunk_type': 'mlp',
            'hidden_sizes': [512]
        }
    }

    opt = {
        'kwargs': {
            'lr': 5e-4
        }
    }


add_params = ex.capture(add_params)


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = CategoricalHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


@ex.automain
def main(_seed, _run):
    torch.manual_seed(_seed)

    env = gym.make('CartPole-v0')
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent_params = PolicyGradientsAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')
    agent_params.sacred_run = _run
    agent_params.env = env
    agent_params.mode = 'train'
    agent_params.reward_filter = RewardRescaleFilter(200.)

    policy_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_net = policy_net

    agent = agent_params.make_agent()
    agent.run()
