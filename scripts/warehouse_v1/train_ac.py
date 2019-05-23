import sacred
import torch
import torch.nn as nn
import glob
import os

from graphrl.sacred.config import add_params, maybe_add_slack
from graphrl.agents.actor_critic_agent import ActorCriticAgentParams
from graphrl.modules.heads import CategoricalValueHead
from graphrl.modules.trunks import build_trunk, ALL_TRUNK_CONFIG
from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env
from graphrl.environments.wrappers import RenderEnv, SampleEnv


ex = sacred.Experiment('train_warehouse_v1_ac')
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
        'policy_entropy_weight': 0.01,
        'update_freq_counter': 'steps',
        'update_freq_steps': 5,
        'max_steps_per_train_episode': 100,
        'max_steps_per_test_episode': 100,
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


add_params = ex.capture(add_params)


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
    head = CategoricalValueHead(trunk_output_size, num_actions)
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


@ex.automain
def main(_seed, _run, env):
    torch.manual_seed(_seed)

    train_envs = build_envs(**env['train'])
    train_env = SampleEnv(train_envs)

    test_envs = build_envs(**env['test'])

    input_shape = train_env.observation_space.shape
    num_actions = train_env.action_space.n

    agent_params = ActorCriticAgentParams()
    add_params(params=agent_params, prefix='agent')
    add_params(params=agent_params.optimizer_params, prefix='opt')

    agent_params.sacred_run = _run
    agent_params.train_env = train_env
    agent_params.test_envs = test_envs

    policy_value_net = build_net(input_shape=input_shape, num_actions=num_actions)
    agent_params.policy_value_net = policy_value_net

    agent = agent_params.make_agent()
    agent.run()
