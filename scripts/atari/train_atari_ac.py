import gym
import sacred
import torch
import torch.nn as nn

from graphrl.sacred.config import add_params
from graphrl.agents.filters import AtariObservationFilter
from graphrl.agents.actor_critic_agent import ActorCriticAgentParams
from graphrl.modules.heads import CategoricalValueHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.atari_wrappers import wrap_deepmind


ex = sacred.Experiment('train_atari_ac')


add_params = ex.capture(add_params)


@ex.config
def config():
    arch = {
        'trunk': ALL_TRUNK_CONFIG
    }


@ex.config
def config():
    game = 'pong'

    agent = {
        'policy_entropy_weight': 0.05,
        'update_freq_counter': 'steps',
        'update_freq_steps': 20,
    }

    opt = {
        'kwargs': {
            'lr': 1e-4
        }
    }

    arch = {
        'trunk': {
            'trunk_type': 'nature'
        }
    }


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = CategoricalValueHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


def lower_under_to_upper(s):
    s = s.replace('_', ' ')
    s = s.title()
    return s.replace(' ', '')


@ex.automain
def main(game, _seed, _run):
    torch.manual_seed(_seed)

    game = lower_under_to_upper(game) + 'NoFrameskip-v4'
    env = gym.make(game)
    env = wrap_deepmind(env)

    input_space = env.observation_space
    num_actions = env.action_space.n

    agent_params = ActorCriticAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')
    agent_params.sacred_run = _run
    agent_params.env = env
    agent_params.mode = 'train'

    agent_params.obs_filter = AtariObservationFilter()
    input_space = agent_params.obs_filter.output_space(input_space)

    policy_value_net = build_net(input_shape=input_space.shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net

    agent = agent_params.make_agent()
    agent.run()
