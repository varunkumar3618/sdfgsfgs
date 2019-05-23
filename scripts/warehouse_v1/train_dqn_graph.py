import sacred
import gym
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import json

from graphrl.agents.schedule import LinearSchedule, ConstantSchedule
from graphrl.sacred.config import add_params, maybe_add_slack
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.modules.heads import QHead
from graphrl.environments.warehouse.warehouse_v1 import make_warehouse_env, make_warehouse_knowledge_graph
from graphrl.environments.wrappers import RenderEnv, MapEnv, SampleEnv
from graphrl.modules.nn import SpatialReduce, MLP
from graphrl.modules.graph.dense_nn import DynamicGraphConv
from graphrl.modules.graph.dense_sgkg import SGIntoKGConv, KGSGIntoSGConv, KGIntoSGPool
from graphrl.agents.stopping import DoesntTrainStoppingCondition


ex = sacred.Experiment('train_warehouse_dqn_graph')
maybe_add_slack(ex)


@ex.config
def config():
    env = {
        'train': {
            'artfile_folder': 'environments/simple/train',
            'render': False,
            'boxes': ['b'],
            'buckets': ['B'],
            'bucket_to_boxes': [('B', ['b'])],
            'character_map': [],
            'dont_crop_adj': False,
            'no_kg_edges': False,
            'fully_connected_kg': False,
            'fully_connected_distinct_kg': False,
            'use_negative_fills_kg': False,
            'same_edge_feats_kg': False,
            'use_agent_filled_kg': False,
            'use_background_entity_kg': False,
            'should_load_kg': False,
            'load_kg_file': 'kg_train_load.json',
            'should_save_kg': False,
            'save_kg_file': 'kg_train_save.json'
        },
        'test': {
            'artfile_folder': 'environments/simple/train',
            'render': False,
            'boxes': ['b'],
            'buckets': ['B'],
            'bucket_to_boxes': [('B', ['b'])],
            'character_map': [],
            'dont_crop_adj': False,
            'no_kg_edges': False,
            'fully_connected_kg': False,
            'fully_connected_distinct_kg': False,
            'use_background_entity_kg': False,
            'use_negative_fills_kg': False,
            'same_edge_feats_kg': False,
            'use_agent_filled_kg': False,
            'should_load_kg': False,
            'load_kg_file': 'kg_test_load.json',
            'should_save_kg': False,
            'save_kg_file': 'kg_test_save.json'
        }
    }

    agent = {
        'mode': 'train',
        'device_name': 'cuda',
        'replay_buffer_size': 40000,
        'eval_episodes': 2,
        'heatup_steps': 1000,
        'update_freq_steps': 1,
        'update_target_weights_freq_steps': 100,
        'max_steps_per_train_episode': 100,
        'max_steps_per_test_episode': 100,
        'test_epsilon': 0.01,
        'gamma': 0.96,
        'test_episodes': 1000,
        'print_actions': False,

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
            'fc_hidden_sizes': [50],
            'conv_hidden_size': 64,
            'use_sg_into_kg_skip': False,
            'dgconv': {
                'wnet_hidden_size': 8,
                'wnet_num_layers': 1,
                'dynamic_bias': False,
                'use_skip': True,
                'use_reverse_edges': False
            }
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


class GraphTrunk(nn.Module):
    def __init__(self, input_shape, fc_hidden_sizes, num_entities, num_node_feats, num_edge_feats, conv_hidden_size, use_sg_into_kg_skip,
                 dgconv):
        super(GraphTrunk, self).__init__()
        self.num_entities = num_entities
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.conv_hidden_size = conv_hidden_size
        self.use_sg_into_kg_skip = use_sg_into_kg_skip

        dgconv = dict(dgconv)
        wnet_layers = [dgconv.pop('wnet_hidden_size')] * dgconv.pop('wnet_num_layers')

        self.kg_dgconv1 = DynamicGraphConv(self.num_node_feats, self.conv_hidden_size, self.num_edge_feats, wnet_layers, **dgconv)
        self.kg_dgconv2 = DynamicGraphConv(self.conv_hidden_size, self.conv_hidden_size, self.num_edge_feats, wnet_layers, **dgconv)

        self.kg_sg_pool = KGIntoSGPool()

        self.sg_conv1 = KGSGIntoSGConv(sg_in_feats=self.conv_hidden_size, sg_out_feats=self.conv_hidden_size, kg_in_feats=self.conv_hidden_size)

        self.sg_kg_conv1 = SGIntoKGConv(sg_in_feats=self.conv_hidden_size,
                                        kg_out_feats=self.conv_hidden_size,
                                        kg_in_feats=self.conv_hidden_size,
                                        use_skip=self.use_sg_into_kg_skip)

        self.kg_dgconv3 = DynamicGraphConv(self.conv_hidden_size, self.conv_hidden_size, self.num_edge_feats, wnet_layers, **dgconv)
        self.kg_dgconv4 = DynamicGraphConv(self.conv_hidden_size, self.conv_hidden_size, self.num_edge_feats, wnet_layers, **dgconv)

        self.sg_conv2 = KGSGIntoSGConv(sg_in_feats=self.conv_hidden_size, sg_out_feats=self.conv_hidden_size, kg_in_feats=self.conv_hidden_size)

        self.spatial_reduce = SpatialReduce(reduction='mean')
        self.mlp = MLP(self.conv_hidden_size, fc_hidden_sizes, nn.ReLU)

    def compute_sg_kg_adj(self, obs, num_entities):
        bz, height, width = obs.shape
        device = obs.device

        obs = obs.view(bz, height * width)[:, :, None]
        sg_kg_adj = obs.new_zeros((bz, height * width, num_entities), device=device, dtype=torch.float32)
        sg_kg_adj.scatter_(2, obs, 1)
        return sg_kg_adj

    def forward(self, batch):
        kg_node_feats, kg_adj, kg_rel, obs = batch
        sg_kg_adj = self.compute_sg_kg_adj(obs, int(kg_adj.size(1)))

        # Graph conv on kg
        kg_node_feats = F.relu(self.kg_dgconv1(kg_node_feats, kg_adj, kg_rel))
        kg_node_feats = F.relu(self.kg_dgconv2(kg_node_feats, kg_adj, kg_rel))

        # Pool into sg
        sg_node_feats = self.kg_sg_pool(kg_node_feats, obs)

        # sg/kg into sg conv
        sg_node_feats = F.relu(self.sg_conv1(obs=obs, sg_feats=sg_node_feats, kg_feats=kg_node_feats))

        # sg/kg into kg conv
        kg_node_feats = F.relu(self.sg_kg_conv1(sg_kg_adj=sg_kg_adj, sg_feats=sg_node_feats, kg_feats=kg_node_feats))

        # Graph conv on kg
        kg_node_feats = F.relu(self.kg_dgconv3(kg_node_feats, kg_adj, kg_rel))
        kg_node_feats = F.relu(self.kg_dgconv4(kg_node_feats, kg_adj, kg_rel))

        # sg/kg into sg conv
        sg_node_feats = F.relu(self.sg_conv2(obs=obs, sg_feats=sg_node_feats, kg_feats=kg_node_feats))

        x = self.spatial_reduce(sg_node_feats)
        x = self.mlp(x)
        return x


def build_trunk(input_shape, fc_hidden_sizes, num_entities, num_node_feats, num_edge_feats, **kwargs):
    trunk = GraphTrunk(input_shape=input_shape, num_entities=num_entities,
                       fc_hidden_sizes=fc_hidden_sizes, num_node_feats=num_node_feats, num_edge_feats=num_edge_feats,
                       **kwargs)
    return trunk, fc_hidden_sizes[-1]


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, num_entities, num_node_feats, num_edge_feats, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape,
                                           num_entities=num_entities,
                                           num_node_feats=num_node_feats,
                                           num_edge_feats=num_edge_feats,
                                           **trunk)
    head = QHead(trunk_output_size, num_actions)
    return nn.Sequential(trunk, head)


class TorchList(object):
    def __init__(self, elems, device=None):
        super(TorchList, self).__init__()
        if device is not None:
            elems = [elem.to(device) for elem in elems]
        elif len(elems) > 0:
            device = elems[0].device
        self.device = device
        self.elems = elems

    def __len__(self):
        return len(self.elems)

    def __getitem__(self, idx):
        return self.elems[idx]

    def to(self, device):
        if device == self.device:
            return self
        else:
            return TorchList(self.elems, device=device)


class GraphEnv(gym.ObservationWrapper):
    def __init__(self, env, KG, num_entities, num_node_feats, num_edge_feats, dont_crop_adj):
        super(GraphEnv, self).__init__(env)

        self.KG = KG
        self.num_entities = num_entities
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.dont_crop_adj = dont_crop_adj

        self.kg_adj = nx.to_numpy_array(self.KG).astype(np.float32)

        kg_rel = np.zeros((self.num_entities, self.num_entities, self.num_edge_feats), dtype=np.float32)
        for i, j in self.KG.edges:
            kg_rel[i, j] = self.KG.edges[i, j]['edge_feature']
        self.kg_rel = kg_rel

        self.kg_node_feats = np.stack((self.KG.nodes[i]['node_feature'] for i in self.KG.nodes))

    def observation(self, obs):
        if self.dont_crop_adj:
            obs_vals_list = list(range(self.num_entities))
        else:
            obs_vals_list = list(sorted(list(set(obs.flatten()))))
        obs_vals = np.array(obs_vals_list)

        kg_node_feats = self.kg_node_feats[obs_vals]
        kg_adj = self.kg_adj[obs_vals[:, None], obs_vals[None, :]]
        kg_rel = self.kg_rel[obs_vals[:, None], obs_vals[None, :]]

        obs = np.vectorize(obs_vals_list.index)(obs)

        return torch.from_numpy(kg_node_feats), torch.from_numpy(kg_adj), torch.from_numpy(kg_rel), torch.from_numpy(obs)

    @staticmethod
    def batch_observations(obs_batch):
        kg_node_feats_list, kg_adj_list, kg_rel_list, obs_list = zip(*obs_batch)

        max_num_entities = max(int(kg_node_feats.size(0)) for kg_node_feats in kg_node_feats_list)

        kg_node_feats_list = [F.pad(kg_node_feats, (0, 0, 0, max_num_entities - kg_node_feats.size(0))) for kg_node_feats in kg_node_feats_list]
        kg_adj_list = [F.pad(kg_adj, (0, max_num_entities - kg_adj.size(0), 0, max_num_entities - kg_adj.size(0)), 'constant', 0)
                       for kg_adj in kg_adj_list]
        kg_rel_list = [F.pad(kg_rel, (0, 0, 0, max_num_entities - kg_rel.size(0), 0, max_num_entities - kg_rel.size(0)), 'constant', 0)
                       for kg_rel in kg_rel_list]

        tensors = [kg_node_feats_list, kg_adj_list, kg_rel_list, obs_list]
        tensors = [torch.stack(t, 0) for t in tensors]
        return TorchList(tensors)


def build_envs(artfile_folder, boxes, buckets, bucket_to_boxes, character_map={}, encode_onehot=False, render=False,
               dont_crop_adj=False, no_kg_edges=False, fully_connected_kg=False, fully_connected_distinct_kg=False, use_negative_fills_kg=False, use_background_entity_kg=False,
               same_edge_feats_kg=False, use_agent_filled_kg=False, should_load_kg=False, load_kg_file=None, should_save_kg=False, save_kg_file=None):
    artfiles = glob.glob(os.path.join(artfile_folder, '*.txt'))

    print('Reading artfile folder: {}'.format(artfile_folder))
    for i, artfile in enumerate(artfiles):
        print('{}) {}'.format(i + 1, artfile))

    bucket_to_boxes = dict(bucket_to_boxes)
    character_map = dict(character_map)
    kg_entities, KG, num_node_feats, num_edge_feats = make_warehouse_knowledge_graph(boxes=boxes, buckets=buckets, bucket_to_boxes=bucket_to_boxes, character_map=character_map,
                                                                                     no_edges=no_kg_edges, fully_connected=fully_connected_kg, fully_connected_distinct=fully_connected_distinct_kg,
                                                                                     use_negative_fills=use_negative_fills_kg, use_background_entity=use_background_entity_kg,
                                                                                     same_edge_feats=same_edge_feats_kg,
                                                                                     use_agent_filled=use_agent_filled_kg)

    if should_load_kg:
        with open(load_kg_file, 'r') as f:
            load_d = json.load(f)

        kg_entities = [ord(c) for c in load_d['kg_entities']]

        KG = nx.DiGraph()
        for node_d in load_d['nodes']:
            feature = np.zeros((node_d['feature_len'],), dtype=np.float32)
            feature[node_d['feature_idx']] = 1
            KG.add_node(ord(node_d['node']), node_feature=feature)

            num_node_feats = node_d['feature_len']

        for edge_d in load_d['edges']:
            feature = np.zeros((edge_d['feature_len'],), dtype=np.float32)
            feature[edge_d['feature_idx']] = 1
            KG.add_edge(ord(edge_d['src']), ord(edge_d['dst']), edge_feature=feature)

            num_edge_feats = edge_d['feature_len']

    save_d = {}
    save_d['kg_entities'] = [chr(v) for v in kg_entities]
    save_d['nodes'] = []
    save_d['edges'] = []

    for x in KG.nodes:
        node_feature = KG.nodes[x]['node_feature']
        node_d = {}
        node_d['node'] = chr(x)
        node_d['feature_idx'] = int(node_feature.argmax())
        node_d['feature_len'] = len(node_feature)
        save_d['nodes'].append(node_d)

    for x, y in KG.edges:
        edge_feature = KG.edges[x, y]['edge_feature']
        edge_d = {}
        edge_d['src'] = chr(x)
        edge_d['dst'] = chr(y)
        edge_d['feature_idx'] = int(edge_feature.argmax())
        edge_d['feature_len'] = len(edge_feature)
        save_d['edges'].append(edge_d)

    print(json.dumps(save_d, indent=4))

    if should_save_kg:
        with open(save_kg_file, 'w') as f:
            json.dump(save_d, f, indent=4)

    KG = nx.relabel_nodes(KG, {entity: i for i, entity in enumerate(kg_entities)})

    print('KG entities: {}'.format([chr(entity) for entity in kg_entities]))
    print('KG node num feats: {}'.format(num_node_feats))
    print('KG edge num feats: {}'.format(num_edge_feats))
    print('KG')
    print('KG nodes')
    for node in KG.nodes:
        print('Node: {}, character: {}, node feature length: {}, node feature argmax: {}'.format(node, chr(kg_entities[node]), len(KG.nodes[node]['node_feature']), int(KG.nodes[node]['node_feature'].argmax())))
    print('KG edges')
    for src, dst in KG.edges:
        print('Source: {}, dst: {}, src entity: {}, dst entity: {}, edge feature length: {}, edge feature {}'.format(src, dst, chr(kg_entities[src]), chr(kg_entities[dst]), len(KG.edges[src, dst]['edge_feature']), int(KG.edges[src, dst]['edge_feature'].argmax())))

    def env_func(artfile):
        env = make_warehouse_env(artfile,
                                 boxes=boxes,
                                 buckets=buckets,
                                 bucket_to_boxes=bucket_to_boxes,
                                 character_map=character_map,
                                 encode_onehot=encode_onehot)
        env = MapEnv(env, {entity: i for i, entity in enumerate(kg_entities)})
        env = GraphEnv(env=env, KG=KG,
                       num_entities=len(kg_entities),
                       num_node_feats=num_node_feats,
                       num_edge_feats=num_edge_feats,
                       dont_crop_adj=dont_crop_adj)
        if render:
            env = RenderEnv(env)
        return env

    envs = [env_func(artfile) for artfile in artfiles]
    return envs, (kg_entities, KG, num_node_feats, num_edge_feats)


@ex.capture(prefix='eps')
def add_epsilon_params(params, eps_type, constant_value, initial_value, final_value, decay_steps):
    if eps_type == 'linear':
        params.train_epsilon_schedule = LinearSchedule(initial_value, final_value, decay_steps)
    elif eps_type == 'constant':
        params.train_epsilon_schedule = ConstantSchedule(constant_value)


def batch_fn(elem_list):
    elem_list = zip(*elem_list)
    elem_list = [torch.stack(elems, 0) for elems in elem_list]
    return TorchList(elem_list)


@ex.capture(prefix='stop')
def add_stopping_params(params, min_episodes, doesnt_train_episodes, bad_reward):
    stopping_cond = DoesntTrainStoppingCondition(min_episodes, doesnt_train_episodes, bad_reward)
    params.custom_stopping_cond = stopping_cond


@ex.automain
def main(_seed, _run, env):
    torch.manual_seed(_seed)

    train_envs, (kg_entities, _, num_node_feats, num_edge_feats) = build_envs(**env['train'])
    num_entities = len(kg_entities)
    train_env = SampleEnv(train_envs)

    test_envs, _ = build_envs(**env['test'])

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

    agent_params.obs_filter = GraphEnv.batch_observations

    online_q_net = build_net(input_shape=input_shape, num_actions=num_actions,
                             num_entities=num_entities, num_node_feats=num_node_feats, num_edge_feats=num_edge_feats)
    target_q_net = build_net(input_shape=input_shape, num_actions=num_actions,
                             num_entities=num_entities, num_node_feats=num_node_feats, num_edge_feats=num_edge_feats)
    agent_params.online_q_net = online_q_net
    agent_params.target_q_net = target_q_net

    agent = agent_params.make_agent()
    agent.run()
