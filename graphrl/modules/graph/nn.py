import collections
import math
import numpy as np
import torch
import torch.nn as nn
import dgl.function as dgl_fn


def graph_map_to_conv_map(tensor, height, width, channels):
    bz = tensor.size(0)
    tensor = tensor.view(bz, height, width, channels)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def conv_map_to_graph_map(tensor):
    bz, channels, height, width = tensor.shape
    tensor = tensor.permute(0, 2, 3, 1).contiguous()
    tensor = tensor.view(bz, height * width, channels)
    return tensor


def get_default_indices(input_features, in_start_idx, in_end_idx, out_start_idx, out_end_idx):
    if in_start_idx is None:
        in_start_idx = 0
    if in_end_idx is None:
        in_end_idx = input_features.shape[1]
    if out_start_idx is None:
        out_start_idx = 0
    if out_end_idx is None:
        out_end_idx = input_features.shape[1]
    return in_start_idx, in_end_idx, out_start_idx, out_end_idx


def compute_indices(batch_num_nodes, start_idx, end_idx):
    indices = []
    offset = 0
    for num_nodes in batch_num_nodes:
        indices.extend(range(offset + start_idx, offset + end_idx))
        offset += num_nodes
    return indices


def set_input_data(gbatch, field, input_features, in_start_idx, in_end_idx):
    bz = gbatch.batch_size
    input_indices = compute_indices(gbatch.batch_num_nodes, in_start_idx, in_end_idx)
    gbatch.nodes[input_indices].data[field] = input_features.view(bz * (in_end_idx - in_start_idx), -1)


def get_output_data(gbatch, field, out_start_idx, out_end_idx):
    bz = gbatch.batch_size
    output_features = gbatch.ndata.pop(field)
    output_indices = compute_indices(gbatch.batch_num_nodes, out_start_idx, out_end_idx)
    output_features = output_features[output_indices]
    output_features = output_features.view(bz, (out_end_idx - out_start_idx), -1)
    return output_features


class GraphConvApplyModule(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, bias):
        super(GraphConvApplyModule, self).__init__()
        self.fc = nn.Linear(num_in_feats, num_out_feats, bias=bias)

    def forward(self, node):
        h = self.fc(node.data['h'])
        return {'h': h}


class GraphConv(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, bias=True):
        super(GraphConv, self).__init__()
        self.apply_mod = GraphConvApplyModule(num_in_feats, num_out_feats, bias=bias)
        self.msg_fn = dgl_fn.copy_src(src='h', out='m')
        self.red_fn = dgl_fn.sum(msg='m', out='h')

    def forward(self, input_features, gbatch, in_start_idx=None, in_end_idx=None, out_start_idx=None, out_end_idx=None):
        in_start_idx, in_end_idx, out_start_idx, out_end_idx = get_default_indices(input_features, in_start_idx, in_end_idx, out_start_idx, out_end_idx)

        set_input_data(gbatch, 'h', input_features, in_start_idx, in_end_idx)

        gbatch.update_all(self.msg_fn, self.red_fn)
        gbatch.apply_nodes(func=self.apply_mod)

        output_features = get_output_data(gbatch, 'h', out_start_idx, out_end_idx)
        return output_features


class WeightNet(nn.Sequential):
    """ Weight generating network. """

    def __init__(self, hidden_units, in_shape, out_shape, name='wnet'):
        self.name = name

        # needed to cat output to correct shape
        self.out_shape = out_shape

        layers = []
        in_feats = int(np.prod(in_shape))
        for ix, out_feats in enumerate(hidden_units):
            layers.append((name + '_lin%s' % ix, nn.Linear(in_feats, out_feats)))
            layers.append((name + '_relu%s' % ix, nn.ReLU(inplace=True)))
            in_feats = out_feats

        # final layer doesn't have ReLu (weights are positive and negative)
        weight_size = int(np.prod(out_shape))
        layers.append((name + '_weightlin', nn.Linear(in_feats, weight_size)))

        super().__init__(collections.OrderedDict(layers))

    def forward(self, input):
        bsize = input.shape[0]
        out = super().forward(input.view(bsize, -1))
        return out.view(-1, *self.out_shape)


class DynamicGraphConv(nn.Module):
    def __init__(self, num_in_feats, num_out_feats, num_edge_in_feats, wnet_hidden_units, bias=True, reduction=torch.mean):
        super(DynamicGraphConv, self).__init__()
        self.num_in_feats = num_in_feats
        self.num_out_feats = num_out_feats
        self.num_edge_in_feats = num_edge_in_feats
        self.reduction = reduction

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_out_feats,))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.Tensor(self.num_out_feats, self.num_in_feats))
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        def red_fn(nodes):
            return {'h': self.reduction(nodes.mailbox['m'], dim=1)}
        self.red_fn = red_fn
        self.wnet = WeightNet(wnet_hidden_units, (self.num_edge_in_feats,), (self.num_in_feats, self.num_out_feats))

    def forward(self, input_features, gbatch, edge_feature_name, in_start_idx=None, in_end_idx=None, out_start_idx=None, out_end_idx=None):
        in_start_idx, in_end_idx, out_start_idx, out_end_idx = get_default_indices(input_features, in_start_idx, in_end_idx, out_start_idx, out_end_idx)

        set_input_data(gbatch, 'h', input_features, in_start_idx, in_end_idx)

        def msg_fn(edges):
            edge_features = edges.data[edge_feature_name]
            weights = self.wnet(edge_features)
            msg = torch.bmm(edges.src['h'].unsqueeze(1), weights).squeeze(1)
            return {'m': msg}

        gbatch.update_all(msg_fn, self.red_fn)

        output_features = get_output_data(gbatch, 'h', out_start_idx, out_end_idx)

        if self.bias is not None:
            output_features = output_features + self.bias[None, None, :]

        return output_features


class GraphPooling(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(GraphPooling, self).__init__()
        self.reduction = reduction

    def forward(self, input_features, gbatch, in_start_idx=None, in_end_idx=None, out_start_idx=None, out_end_idx=None):
        in_start_idx, in_end_idx, out_start_idx, out_end_idx = get_default_indices(input_features, in_start_idx, in_end_idx, out_start_idx, out_end_idx)

        set_input_data(gbatch, 'h', input_features, in_start_idx, in_end_idx)

        msg_fn = dgl_fn.copy_src(src='h', out='m')

        def red_fn(nodes):
            return {'h': self.reduction(nodes.mailbox['m'], dim=1)}
        gbatch.update_all(msg_fn, red_fn)

        output_features = get_output_data(gbatch, 'h', out_start_idx, out_end_idx)
        return output_features
