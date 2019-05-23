import torch.optim as optim


class OptimizerParams(object):
    def __init__(self):
        self.optimizer_name = 'Adam'
        self.kwargs = {}

    def make_optimizer(self, net):
        return getattr(optim, self.optimizer_name)(net.parameters(), **self.kwargs)

    def __str__(self):
        message = self.optimizer_name
        message += '{'
        for k, v in self.kwargs.items():
            message += '{}:{}'.format(k, v)
        message += '}'
        return message
