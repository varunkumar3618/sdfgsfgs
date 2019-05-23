import collections
import math
import numpy as np
import torch
import torch.nn as nn


class SGIntoKGPool(nn.Module):
    def forward(self, sg_node_feats, sg_kg_adj):
        bz, channels, height, width = sg_node_feats.shape

        sg_node_feats = sg_node_feats.view(bz, channels, height * width)[:, :, :, None]

        degrees = sg_kg_adj.sum(1)
        norm = torch.max(degrees, torch.ones_like(degrees))

        message_feats = sg_node_feats * sg_kg_adj[:, None, :, :]
        output_feats = message_feats.sum(2) / norm[:, None, :]
        output_feats = output_feats.permute(0, 2, 1)

        return output_feats


class SGIntoKGConv(nn.Module):
    def __init__(self, sg_in_feats, kg_out_feats, kg_in_feats=None, use_skip=False):
        super(SGIntoKGConv, self).__init__()
        self.use_skip = use_skip
        self.sg_in_feats = sg_in_feats
        self.kg_out_feats = kg_out_feats
        self.kg_in_feats = kg_in_feats

        self.sg_conv = nn.Conv2d(sg_in_feats, kg_out_feats, 1)
        if self.use_skip:
            self.kg_fc = nn.Linear(kg_in_feats, kg_out_feats)
        self.pool = SGIntoKGPool()

    def forward(self, sg_kg_adj, sg_feats, kg_feats=None):
        sg_feats = self.sg_conv(sg_feats)
        sg_into_kg_feats = self.pool(sg_feats, sg_kg_adj)
        if self.use_skip:
            kg_feats = self.kg_fc(kg_feats)
            sg_into_kg_feats = sg_into_kg_feats + kg_feats
        return sg_into_kg_feats


class KGIntoSGPool(nn.Module):
    def forward(self, kg_node_feats, obs):
        bz, height, width = obs.shape
        channels = kg_node_feats.size(2)

        sg_node_feats = torch.gather(kg_node_feats, 1, obs.view(bz, height * width, 1).repeat(1, 1, channels))
        sg_node_feats = sg_node_feats.view(bz, height, width, channels)
        sg_node_feats = sg_node_feats.permute(0, 3, 1, 2)
        return sg_node_feats


class KGSGIntoSGConv(nn.Module):
    def __init__(self, sg_in_feats, sg_out_feats, kg_in_feats):
        super(KGSGIntoSGConv, self).__init__()
        self.sg_in_feats = sg_in_feats
        self.sg_out_feats = sg_out_feats
        self.kg_in_feats = kg_in_feats

        self.kg_fc = nn.Linear(self.kg_in_feats, self.sg_out_feats)
        self.sg_conv = nn.Conv2d(sg_in_feats, sg_out_feats, 3, padding=1)
        self.pool = KGIntoSGPool()

    def forward(self, obs, sg_feats, kg_feats):
        kg_into_sg_feats = self.pool(self.kg_fc(kg_feats), obs)
        sg_feats = self.sg_conv(sg_feats)
        kg_into_sg_feats = kg_into_sg_feats + sg_feats
        return kg_into_sg_feats
