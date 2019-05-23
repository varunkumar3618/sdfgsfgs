import numpy as np
import torch.nn as nn

from graphrl.modules.nn import Lambda, BatchFlatten, MLP, ConvStack, SpatialReduce


class MLPTrunk(nn.Module):
    def __init__(self, input_shape, hidden_sizes, activation):
        super(MLPTrunk, self).__init__()
        layers = []
        layers.append(Lambda(lambda x: x.float()))
        layers.append(BatchFlatten())
        layers.append(MLP(int(np.prod(input_shape)), hidden_sizes, activation))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class ConvMLPTrunk(nn.Module):
    def __init__(self,
                 input_shape,
                 conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides,
                 fc_hidden_sizes,
                 activation):
        super(ConvMLPTrunk, self).__init__()
        self.conv_stack = ConvStack(input_shape[0], conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides, activation)
        conv_output_shape = self.conv_stack.compute_output_shape(input_shape)
        self.mlp_trunk = MLPTrunk(conv_output_shape, fc_hidden_sizes, activation)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.mlp_trunk(x)
        return x


NATURE_FC_HIDDEN_SIZE = 512


class NatureTrunk(nn.Module):
    def __init__(self, input_shape, activation):
        super(NatureTrunk, self).__init__()
        permuted_shape = (input_shape[2], input_shape[0], input_shape[1])
        self.conv_mlp_trunk = ConvMLPTrunk(input_shape=permuted_shape,
                                           conv_out_cs=[32, 64, 64], conv_filter_sizes=[8, 4, 3], conv_paddings=[0, 0, 0], conv_strides=[4, 2, 1],
                                           fc_hidden_sizes=[NATURE_FC_HIDDEN_SIZE],
                                           activation=activation)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = x / 255.
        return self.conv_mlp_trunk(x)


class ConvReduceMLPTrunk(nn.Module):
    def __init__(self,
                 input_shape,
                 conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides,
                 fc_hidden_sizes,
                 activation,
                 reduction):
        super(ConvReduceMLPTrunk, self).__init__()
        seq = []

        conv_stack = ConvStack(input_shape[0], conv_out_cs, conv_filter_sizes, conv_paddings, conv_strides, activation)
        seq.append(conv_stack)

        seq.append(SpatialReduce(reduction))

        mlp_trunk = MLPTrunk(conv_out_cs[-1], fc_hidden_sizes, activation)
        seq.append(mlp_trunk)

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


ALL_TRUNK_CONFIG = {
    'trunk_type': '',
    'hidden_sizes': [],
    'conv_out_cs': [], 'conv_filter_sizes': [], 'conv_paddings': [], 'conv_strides': [],
    'fc_hidden_sizes': [],
    'reduction': ''
}


# For use with sacred
def build_trunk(input_shape, activation=nn.ReLU, **kwargs):
    trunk_type = kwargs['trunk_type']
    if trunk_type == 'mlp':
        trunk = MLPTrunk(input_shape, kwargs['hidden_sizes'], activation)
        output_size = kwargs['hidden_sizes'][-1]
    elif trunk_type == 'conv':
        trunk = ConvMLPTrunk(input_shape,
                             kwargs['conv_out_cs'], kwargs['conv_filter_sizes'], kwargs['conv_paddings'], kwargs['conv_strides'],
                             kwargs['fc_hidden_sizes'],
                             activation)
        output_size = kwargs['fc_hidden_sizes'][-1]
    elif trunk_type == 'conv_reduce':
        trunk = ConvReduceMLPTrunk(input_shape,
                                   kwargs['conv_out_cs'], kwargs['conv_filter_sizes'], kwargs['conv_paddings'], kwargs['conv_strides'],
                                   kwargs['fc_hidden_sizes'],
                                   activation, kwargs['reduction'])
        output_size = kwargs['fc_hidden_sizes'][-1]
    elif trunk_type == 'nature':
        trunk = NatureTrunk(input_shape, activation)
        output_size = NATURE_FC_HIDDEN_SIZE
    else:
        raise ValueError('Unrecognized trunk type {}'.format(trunk_type))
    return trunk, output_size
