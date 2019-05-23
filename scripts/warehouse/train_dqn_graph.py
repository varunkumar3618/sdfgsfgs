import sacred
import gym
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from graphrl.agents.schedule import LinearSchedule, ConstantSchedule
from graphrl.sacred.config import add_params
from graphrl.agents.deep_q_agent import DeepQAgentParams
from graphrl.modules.heads import QHead
from graphrl.environments.warehouse.warehouse_v0 import make_sampled_warehouse_env, make_warehouse_knowledge_graph
from graphrl.environments.wrappers import RenderEnv, MapEnv
from graphrl.modules.graph.nn import GraphPooling, graph_map_to_conv_map, conv_map_to_graph_map, DynamicGraphConv
from graphrl.modules.nn import SpatialReduce, MLP


ex = sacred.Experiment('train_warehouse_dqn_graph')


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
            'fc_hidden_sizes': [50],
            'height': 10,
            'width': 10,
            'conv_hidden_size': 64
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


class GraphTrunk(nn.Module):
    def __init__(self, input_shape, fc_hidden_sizes, KG, num_entities, num_node_feats, num_edge_feats, height, width, conv_hidden_size):
        super(GraphTrunk, self).__init__()
        self.KG = KG
        self.num_entities = num_entities
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.height = height
        self.width = width
        self.conv_hidden_size = conv_hidden_size

        self.mean_pooling = GraphPooling()

        self.kg_dgconv1 = DynamicGraphConv(self.num_node_feats, self.conv_hidden_size, self.num_edge_feats, [8])
        self.kg_dgconv2 = DynamicGraphConv(self.conv_hidden_size, self.conv_hidden_size, self.num_edge_feats, [8])
        self.kg_dgconv3 = DynamicGraphConv(self.conv_hidden_size, self.conv_hidden_size, self.num_edge_feats, [8])
        self.kg_dgconv4 = DynamicGraphConv(self.conv_hidden_size, self.conv_hidden_size, self.num_edge_feats, [8])

        self.sg_conv1 = nn.Conv2d(self.conv_hidden_size, self.conv_hidden_size, 3, padding=1)
        self.kg_proj1 = nn.Linear(self.conv_hidden_size, self.conv_hidden_size, bias=False)

        self.sg_conv2 = nn.Conv2d(self.conv_hidden_size, self.conv_hidden_size, 3, padding=1)
        self.kg_proj2 = nn.Linear(self.conv_hidden_size, self.conv_hidden_size, bias=False)

        self.spatial_reduce = SpatialReduce(reduction='mean')
        self.mlp = MLP(self.conv_hidden_size, fc_hidden_sizes, nn.ReLU)

    def sg_node_feats_conv(self, layer, sg_node_feats, channels):
        sg_node_feats = layer(graph_map_to_conv_map(sg_node_feats, self.height, self.width, channels))
        sg_node_feats = conv_map_to_graph_map(sg_node_feats)
        return sg_node_feats

    def pool_sg_into_kg(self, sg_node_feats, SG_KG_dgl):
        bz, channels = sg_node_feats.size(0), sg_node_feats.size(1)
        sg_node_feats = sg_node_feats.view(bz, channels, self.height * self.width).permute(0, 2, 1).contiguous()
        kg_node_feats = self.mean_pooling(sg_node_feats, SG_KG_dgl,
                                          in_start_idx=0, in_end_idx=self.height * self.width,
                                          out_start_idx=self.height * self.width, out_end_idx=self.height * self.width + self.num_entities)
        return kg_node_feats

    def pool_kg_into_sg(self, kg_node_feats, sg_entities):
        bz, nkg, channels = kg_node_feats.shape
        kg_node_feats = kg_node_feats.permute(0, 2, 1)
        sg_entities = sg_entities[:, None, :].repeat(1, channels, 1)
        sg_node_feats = torch.gather(kg_node_feats, 2, sg_entities)
        sg_node_feats = sg_node_feats.view(bz, channels, self.height, self.width)
        return sg_node_feats

    def forward(self, batch):
        kg_node_feats, sg_entities, graph_batch = batch

        # Graph conv on kg

        kg_node_feats = F.relu(self.kg_dgconv1(kg_node_feats, graph_batch['KG'], 'edge_feature'))
        kg_node_feats = F.relu(self.kg_dgconv2(kg_node_feats, graph_batch['KG'], 'edge_feature'))

        # Pool into sg

        sg_node_feats = self.pool_kg_into_sg(kg_node_feats, sg_entities)

        # Combined kg/sg conv into sg
        sg_node_feats_conv = self.sg_conv1(sg_node_feats)
        kg_proj_feats = self.pool_kg_into_sg(self.kg_proj1(kg_node_feats), sg_entities)
        sg_node_feats = F.relu(sg_node_feats_conv + kg_proj_feats)

        # Pool into kg
        kg_node_feats = self.pool_sg_into_kg(sg_node_feats, graph_batch['SG_KG'])

        # Graph conv on kg
        kg_node_feats = F.relu(self.kg_dgconv3(kg_node_feats, graph_batch['KG'], 'edge_feature'))
        kg_node_feats = F.relu(self.kg_dgconv4(kg_node_feats, graph_batch['KG'], 'edge_feature'))

        # Combined kg/sg conv into sg
        sg_node_feats_conv = self.sg_conv2(sg_node_feats)
        kg_proj_feats = self.pool_kg_into_sg(self.kg_proj2(kg_node_feats), sg_entities)
        sg_node_feats = F.relu(sg_node_feats_conv + kg_proj_feats)

        x = self.spatial_reduce(sg_node_feats)
        x = self.mlp(x)
        return x


def build_trunk(input_shape, fc_hidden_sizes, **kwargs):
    kg_entities, KG, num_node_feats, num_edge_feats = make_warehouse_knowledge_graph()
    trunk = GraphTrunk(input_shape=input_shape, KG=KG, num_entities=len(kg_entities),
                       fc_hidden_sizes=fc_hidden_sizes, num_node_feats=num_node_feats, num_edge_feats=num_edge_feats,
                       **kwargs)
    return trunk, fc_hidden_sizes[-1]


@ex.capture(prefix='arch')
def build_net(input_shape, num_actions, trunk):
    trunk, trunk_output_size = build_trunk(input_shape=input_shape, **trunk)
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


class GraphBatch(object):
    def __init__(self, graphs, device=None):
        super(GraphBatch, self).__init__()

        if device is not None:
            for key, graph in graphs.items():
                self.graph_to_device(graph, device)

        self.device = device
        self.graphs = graphs

    def __getitem__(self, item):
        return self.graphs[item]

    def __len__(self):
        return len(self.graphs)

    def graph_to_device(self, graph, device):
        for key, tensor in graph.ndata.items():
            graph.ndata[key] = tensor.to(device)
        for key, tensor in graph.edata.items():
            graph.edata[key] = tensor.to(device)

    def to(self, device):
        # This function mutates the current graph.
        if device == self.device:
            return self
        return GraphBatch(self.graphs, device=device)


class GraphEnv(gym.ObservationWrapper):
    def __init__(self, env, KG, num_entities, num_node_feats, num_edge_feats):
        super(GraphEnv, self).__init__(env)

        self.KG = KG
        self.num_entities = num_entities
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.kg_adj = nx.to_numpy_array(self.KG)

        kg_rel = np.zeros((self.num_entities, self.num_entities, self.num_edge_feats), dtype=np.float32)
        for i, j in self.KG.edges:
            kg_rel[i, j] = self.KG.edges[i, j]['edge_feature']
        self.kg_rel = kg_rel

        self.KG_dgl = self.adj_and_rel_to_dgl_same_nodes(self.kg_adj, self.kg_rel)

        self.kg_node_feats = torch.from_numpy(np.stack((self.KG.nodes[i]['node_feature'] for i in self.KG.nodes)))

    def adj_and_rel_to_dgl_same_nodes(self, adj_mtx, rel_mtx=None):
        G_dgl = dgl.DGLGraph()
        G_dgl.add_nodes(adj_mtx.shape[0])
        src, tgt = adj_mtx.nonzero()

        if rel_mtx is None:
            G_dgl.add_edges(src, tgt)
        else:
            edge_features = torch.from_numpy(rel_mtx[src, tgt])
            G_dgl.add_edges(src, tgt, data={'edge_feature': edge_features})
        return G_dgl

    def adj_and_rel_to_dgl_diff_nodes(self, adj_mtx, rel_mtx=None):
        G_dgl = dgl.DGLGraph()
        G_dgl.add_nodes(adj_mtx.shape[0])
        G_dgl.add_nodes(adj_mtx.shape[1])
        src, tgt = adj_mtx.nonzero()

        if rel_mtx is None:
            G_dgl.add_edges(src, tgt + adj_mtx.shape[0])
        else:
            edge_features = torch.from_numpy(rel_mtx[src, tgt])
            G_dgl.add_edges(src, tgt + adj_mtx.shape[0], data={'edge_feature': edge_features})
        return G_dgl

    def observation(self, obs):
        obs = obs.reshape(-1)

        sg_kg_adj = np.zeros((obs.shape[0], self.num_entities), dtype=np.float32)
        sg_kg_adj[range(len(obs)), obs] = 1

        SG_KG_dgl = self.adj_and_rel_to_dgl_diff_nodes(sg_kg_adj)
        KG_SG_dgl = self.adj_and_rel_to_dgl_diff_nodes(sg_kg_adj.T)

        sg_sg_adj = np.dot(np.dot(sg_kg_adj, self.kg_adj), sg_kg_adj.T)

        sg_kg_rel = np.dot(sg_kg_adj, self.kg_rel.reshape(self.num_entities, self.num_entities * self.num_edge_feats))
        sg_kg_rel = sg_kg_rel.reshape(obs.shape[0], self.num_entities, self.num_edge_feats)
        sg_sg_rel = np.dot(np.transpose(sg_kg_rel, (0, 2, 1)), sg_kg_adj.T)
        sg_sg_rel = np.transpose(sg_sg_rel, (0, 2, 1))

        SG_SG_dgl = self.adj_and_rel_to_dgl_same_nodes(sg_sg_adj, sg_sg_rel)

        return self.kg_node_feats, torch.from_numpy(obs), {
            'KG': self.KG_dgl,
            'SG_KG': SG_KG_dgl,
            'KG_SG': KG_SG_dgl,
            'SG_SG': SG_SG_dgl
        }


def list_to_graph_batch(graphs):
    if len(graphs) > 0:
        graphs_dict = {}
        for key in graphs[0].keys():
            vals = [graph[key] for graph in graphs]
            val = dgl.batch(vals)
            graphs_dict[key] = val
        graphs = graphs_dict
    else:
        graphs = {}
    return graphs


def batch_fn(elem_list):
    kg_node_feats_list, sg_entities_list, graphs_list = zip(*elem_list)
    return TorchList([torch.stack(kg_node_feats_list, 0),
                      torch.stack(sg_entities_list, 0),
                      GraphBatch(list_to_graph_batch(graphs_list))])


def build_env(artfiles, render):
    kg_entities, KG, num_node_feats, num_edge_feats = make_warehouse_knowledge_graph()
    env = make_sampled_warehouse_env(artfiles, encode_onehot=False)
    env = MapEnv(env, {entity: i for i, entity in enumerate(kg_entities)})
    env = GraphEnv(env=env, KG=KG,
                   num_entities=len(kg_entities),
                   num_node_feats=num_node_feats,
                   num_edge_feats=num_edge_feats)
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

    agent_params.obs_filter = batch_fn

    agent = agent_params.make_agent()
    agent.run()
