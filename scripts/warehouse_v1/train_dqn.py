import sacred
import torch
import torch.nn as nn
import glob
import os

from graphrl.agents.schedule import LinearSchedule, ConstantSchedule
from graphrl.sacred.config import add_params, maybe_add_slack
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.modules.heads import QHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env
from graphrl.environments.wrappers import RenderEnv, SampleEnv
from graphrl.agents.stopping import DoesntTrainStoppingCondition


ex = sacred.Experiment('train_warehouse_v1_dqn')
maybe_add_slack(ex)


@ex.config
def arch_config():
    arch = {
        'trunk': ALL_TRUNK_CONFIG
    }


@ex.config
def config():
    env = {
        'train': {
            'artfile_folder': 'environments/simple/train',
            'render': False,
            'boxes': ['b'],
            'buckets': ['B'],
            'bucket_to_boxes': [('B', ['b'])],
            'character_map': []
        },
        'test': {
            'artfile_folder': 'environments/simple/train',
            'render': False,
            'boxes': ['b'],
            'buckets': ['B'],
            'bucket_to_boxes': [('B', ['b'])],
            'character_map': []
        }
    }

    agent = {
        'mode': 'train',
        'device_name': 'cuda',
        'replay_buffer_size': 40000,
        'heatup_steps': 1000,
        'eval_episodes': 2,
        'update_freq_steps': 1,
        'update_target_weights_freq_steps': 100,
        'max_steps_per_train_episode': 100,
        'max_steps_per_test_episode': 100,
        'test_epsilon': 0.01,
        'gamma': 0.96,

        'use_min_train_steps': False,
        'use_min_train_episodes': True,
        'min_train_episodes': 40000,

        'use_no_progress_steps': True,
        'no_progress_steps': 50000,

        'should_load_nets': False,
        'load_nets_folder': './model'
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

    stop = {
        'min_episodes': 5000,
        'doesnt_train_episodes': 200,
        'bad_reward': 0
    }


add_params = ex.capture(add_params)


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = QHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


def build_envs(artfile_folder, boxes, buckets, bucket_to_boxes, character_map={}, encode_onehot=True, render=False):
    artfiles = glob.glob(os.path.join(artfile_folder, '*.txt'))

    print('Reading artfile folder: {}'.format(artfile_folder))
    for i, artfile in enumerate(artfiles):
        print('{}) {}'.format(i + 1, artfile))

    bucket_to_boxes = dict(bucket_to_boxes)
    character_map = dict(character_map)

    def env_func(artfile):
        env = make_warehouse_env(artfile,
                                 boxes=boxes,
                                 buckets=buckets,
                                 bucket_to_boxes=bucket_to_boxes,
                                 character_map=character_map,
                                 encode_onehot=encode_onehot)
        if render:
            env = RenderEnv(env)
        return env

    envs = [env_func(artfile) for artfile in artfiles]
    return envs


@ex.capture(prefix='eps')
def add_epsilon_params(params, eps_type, constant_value, initial_value, final_value, decay_steps):
    if eps_type == 'linear':
        params.train_epsilon_schedule = LinearSchedule(initial_value, final_value, decay_steps)
    elif eps_type == 'constant':
        params.train_epsilon_schedule = ConstantSchedule(constant_value)


@ex.capture(prefix='stop')
def add_stopping_params(params, min_episodes, doesnt_train_episodes, bad_reward):
    stopping_cond = DoesntTrainStoppingCondition(min_episodes, doesnt_train_episodes, bad_reward)
    params.custom_stopping_cond = stopping_cond


@ex.automain
def main(_seed, _run, env):
    torch.manual_seed(_seed)

    train_envs = build_envs(**env['train'])
    train_env = SampleEnv(train_envs)

    test_envs = build_envs(**env['test'])

    input_shape = train_env.observation_space.shape
    num_actions = train_env.action_space.n

    agent_params = DeepQAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')
    add_epsilon_params(params=agent_params)
    add_stopping_params(params=agent_params)

    agent_params.sacred_run = _run
    agent_params.train_env = train_env
    agent_params.test_envs = test_envs

    online_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    target_q_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent = agent_params.make_agent()
    agent.run()
