import gym
import sacred
import torch
import torch.nn as nn

from graphrl.agents.filters import RewardRescaleFilter
from graphrl.agents.actor_critic_agent import ActorCriticAgentParams
from graphrl.modules.heads import CategoricalValueHead
from graphrl.modules.trunks import build_trunk
from graphrl.sacred.config import add_params


ex = sacred.Experiment('train_cartpole_ac')


@ex.config
def config():
    agent = {
        'device_name': 'cuda',
        'policy_entropy_weight': 0.01,
        'gae_lambda': 1.,
        'update_freq_counter': 'steps',
        'update_freq_steps': 5
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


@ex.automain
def main(_seed, _run):
    torch.manual_seed(_seed)

    env = gym.make('CartPole-v0')
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent_params = ActorCriticAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')
    agent_params.sacred_run = _run
    agent_params.env = env
    agent_params.mode = 'train'
    agent_params.reward_filter = RewardRescaleFilter(200.)

    policy_value_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net

    agent = agent_params.make_agent()
    agent.run()
