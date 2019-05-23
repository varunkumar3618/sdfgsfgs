import gym
import sacred
import torch
import torch.nn as nn

from graphrl.agents.schedule import LinearSchedule
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.sacred.config import add_params
from graphrl.modules.heads import QHead
from graphrl.modules.trunks import build_trunk


ex = sacred.Experiment('train_cartpole_dqn')


@ex.config
def config():
    agent = {
        'device_name': 'cuda',
        'replay_buffer_size': 40000,
        'heatup_steps': 1000,
        'update_freq_steps': 1,
        'update_target_weights_freq_steps': 100,
        'test_epsilon': 0.01,
        'heatup_steps': 1000
    }

    arch = {
        'trunk': {
            'trunk_type': 'mlp',
            'hidden_sizes': [512]
        }
    }

    opt = {
        'kwargs': {
            'lr': 2.5e-4
        }
    }

    eps = {
        'initial_value': 1.,
        'final_value': 0.01,
        'decay_steps': 10000
    }


add_params = ex.capture(add_params)


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = QHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


@ex.capture(prefix='eps')
def add_epsilon_params(params, initial_value, final_value, decay_steps):
    params.train_epsilon_schedule = LinearSchedule(initial_value, final_value, decay_steps)


@ex.automain
def main(_seed, _run):
    torch.manual_seed(_seed)

    env = gym.make('CartPole-v0')
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent_params = DeepQAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')
    add_epsilon_params(params=agent_params)

    agent_params.sacred_run = _run
    agent_params.env = env
    agent_params.mode = 'train'

    online_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    target_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent = agent_params.make_agent()
    agent.run()
