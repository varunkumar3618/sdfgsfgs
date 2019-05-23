import torch.nn as nn
import torch.distributions as dist

from graphrl.modules.nn import Fork


class CategoricalHead(nn.Module):
    def __init__(self, input_size, num_actions):
        super(CategoricalHead, self).__init__()
        self.fc = nn.Linear(input_size, num_actions)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        logits = self.fc(x)
        return dist.Categorical(logits=logits)


class ValueHead(nn.Module):
    def __init__(self, input_size):
        super(ValueHead, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)[:, 0]


class QHead(nn.Module):
    def __init__(self, input_size, num_actions):
        super(QHead, self).__init__()
        self.fc = nn.Linear(input_size, num_actions)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class CategoricalValueHead(nn.Module):
    def __init__(self, input_size, num_actions):
        super(CategoricalValueHead, self).__init__()
        categorical_head = CategoricalHead(input_size, num_actions)
        value_head = ValueHead(input_size)

        self.fork = Fork(categorical_head, value_head)

    def forward(self, x):
        return self.fork(x)
