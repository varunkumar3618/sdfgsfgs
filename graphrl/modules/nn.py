import torch
import torch.nn as nn


class BatchFlatten(nn.Module):
    def forward(self, x):
        bz = x.size(0)
        return x.view(bz, -1)


class Fork(nn.Module):
    def __init__(self, *args):
        super(Fork, self).__init__()
        self.layers = nn.ModuleList(list(args))

    def forward(self, x):
        results = []
        for i, module in enumerate(self.layers):
            res = module(x)
            results.append(res)
        return tuple(results)


class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation):
        super(MLP, self).__init__()
        seq = []

        for hidden_size in hidden_sizes:
            seq.append(nn.Linear(input_size, hidden_size))
            seq.append(activation())
            input_size = hidden_size
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class SpatialReduce(nn.Module):
    def __init__(self, reduction):
        super(SpatialReduce, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        bz, c = x.size(0), x.size(1)
        x = x.view(bz, c, -1)

        if self.reduction == 'max':
            x = torch.max(x, 2)[0]
        elif self.reduction == 'mean':
            x = torch.mean(x, 2)
        else:
            raise ValueError('Unknown reduction {}.'.format(self.reduction))
        return x


class ConvStack(nn.Module):
    def __init__(self, in_c, out_cs, filter_sizes, paddings, strides, activation):
        super(ConvStack, self).__init__()

        self.out_cs = out_cs
        self.filter_sizes = filter_sizes
        self.paddings = paddings
        self.strides = strides

        seq = []
        for out_c, filter_size, padding, stride in zip(out_cs, filter_sizes, paddings, strides):
            seq.append(nn.Conv2d(in_c, out_c, filter_size, stride=stride, padding=padding))
            seq.append(activation())
            in_c = out_c
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

    def compute_output_shape(self, input_shape):
        for out_c, filter_size, padding, stride in zip(self.out_cs, self.filter_sizes, self.paddings, self.strides):
            c, h, w = input_shape

            h = (h - filter_size + 2 * padding) // stride + 1
            w = (w - filter_size + 2 * padding) // stride + 1

            input_shape = (out_c, h, w)
        return input_shape
