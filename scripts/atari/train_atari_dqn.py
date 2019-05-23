import gym
import sacred
import torch
import torch.nn as nn

from graphrl.sacred.config import add_params
from graphrl.agents.filters import AtariObservationFilter
from graphrl.agents.schedule import LinearSchedule, ConstantSchedule
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.modules.heads import QHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.atari_wrappers import wrap_deepmind


ex = sacred.Experiment('train_atari_dqn')


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
        'device_name': 'cuda',
        'double_dqn': False,
        'update_freq_steps': 1,
        'update_target_weights_freq_steps': 1000,
        'heatup_steps': 10000,
        'use_huber_loss': False,
        'test_epsilon': 0.02,
        'replay_buffer_size': 100000
    }

    opt = {
        'kwargs': {
            'lr': 1e-4
        }
    }

    eps = {
        'eps_type': 'linear',
        'constant_value': 0.02,
        'initial_value': 1.,
        'final_value': 0.02,
        'decay_steps': 100000
    }

    arch = {
        'trunk': {
            'trunk_type': 'nature'
        }
    }


def lower_under_to_upper(s):
    s = s.replace('_', ' ')
    s = s.title()
    return s.replace(' ', '')


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = QHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


@ex.capture(prefix='eps')
def add_epsilon_params(params, eps_type, constant_value, initial_value, final_value, decay_steps):
    if eps_type == 'linear':
        params.train_epsilon_schedule = LinearSchedule(initial_value, final_value, decay_steps)
    elif eps_type == 'constant':
        params.train_epsilon_schedule = ConstantSchedule(constant_value)


@ex.automain
def main(game, _seed, _run):
    torch.manual_seed(_seed)

    game = lower_under_to_upper(game) + 'NoFrameskip-v4'
    env = gym.make(game)
    env = wrap_deepmind(env)

    input_space = env.observation_space
    num_actions = env.action_space.n

    agent_params = DeepQAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')
    add_epsilon_params(params=agent_params)
    agent_params.obs_filter = AtariObservationFilter()

    input_space = agent_params.obs_filter.output_space(input_space)

    agent_params.sacred_run = _run
    agent_params.env = env
    agent_params.mode = 'train'

    online_q_net = build_net(input_shape=input_space.shape, num_actions=num_actions)
    target_q_net = build_net(input_shape=input_space.shape, num_actions=num_actions)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent = agent_params.make_agent()
    agent.run()
