import sacred
import torch
import torch.nn as nn

from graphrl.agents.schedule import LinearSchedule, ConstantSchedule
from graphrl.sacred.config import add_params
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.modules.heads import QHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.warehouse.warehouse_v0 import make_sampled_warehouse_env
from graphrl.environments.wrappers import RenderEnv


ex = sacred.Experiment('train_warehouse_dqn')


@ex.config
def arch_config():
    arch = {
        'trunk': ALL_TRUNK_CONFIG
    }


@ex.config
def config():
    env = {
        'train': {
            'artfiles': ['assets/warehouse_art/10x10/art_10x10_1.txt'],
            'render': False
        },
        'test': {
            'artfiles': ['assets/warehouse_art/10x10/art_10x10_1.txt'],
            'render': False
        }
    }

    agent = {
        'mode': 'train',
        'device_name': 'cuda',
        'replay_buffer_size': 40000,
        'heatup_steps': 1000,
        'update_freq_steps': 1,
        'update_target_weights_freq_steps': 100,
        'max_steps_per_train_episode': 100,
        'max_steps_per_test_episode': 100,
        'test_epsilon': 0.01,
        'gamma': 0.96
    }

    arch = {
        'trunk': {
            'trunk_type': 'mlp',
            'hidden_sizes': [256, 512]
        }
    }

    opt = {
        'kwargs': {
            'lr': 2.5e-4
        }
    }

    eps = {
        'eps_type': 'linear',
        'constant_value': 0.1,
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


def build_env(artfiles, render):
    env = make_sampled_warehouse_env(artfiles, encode_onehot=True)
    if render:
        env = RenderEnv(env)
    return env


@ex.capture(prefix='eps')
def add_epsilon_params(params, eps_type, constant_value, initial_value, final_value, decay_steps):
    if eps_type == 'linear':
        params.train_epsilon_schedule = LinearSchedule(initial_value, final_value, decay_steps)
    elif eps_type == 'constant':
        params.train_epsilon_schedule = ConstantSchedule(constant_value)


@ex.automain
def main(_seed, _run, env):
    torch.manual_seed(_seed)

    train_env, test_env = build_env(**env['train']), build_env(**env['test'])
    input_shape = train_env.observation_space.shape
    num_actions = test_env.action_space.n

    agent_params = DeepQAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')
    add_epsilon_params(params=agent_params)

    agent_params.sacred_run = _run
    agent_params.train_env = train_env
    agent_params.test_envs.append(test_env)

    online_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    target_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent = agent_params.make_agent()
    agent.run()
