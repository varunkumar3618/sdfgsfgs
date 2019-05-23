import collections
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
from torch._six import container_abcs


def make_collate_fn(batchers):
    '''
    Create a collate function for use with a torch dataloader.
    The function will be identical to the torch default except when it sees a dictionary.
    In that case, it will use the custom batchers defined in batchers on known keys. Other keys will be batched using the default.
    NOTE: this function doesn't handle recursive cases. Batchers are only applied at the outer level.
    '''

    def collate_fn(batch):
        if isinstance(batch[0], container_abcs.Mapping):
            new_batch = {}
            keys = set(batch[0].keys())
            for key in keys:
                if key in batchers:
                    values = [elem[key] for elem in batch]
                    values = batchers[key](values)
                else:
                    values = [elem[key] for elem in batch]
                    values = default_collate(values)
                new_batch[key] = values
            return new_batch
        else:
            return default_collate(batch)
    return collate_fn


def compute_returns(rewards, gamma, use_future_return):
    if use_future_return:
        returns = []
        return_val = 0

        for reward in reversed(rewards):
            return_val = return_val * gamma + reward
            returns.append(return_val)
        returns = list(reversed(returns))
        returns = np.array(returns)
    else:
        gamma_vec = np.array([gamma ** i for i in range(len(rewards))])
        return_val = np.sum(gamma_vec * rewards)
        returns = np.repeat(return_val, len(rewards))
    return returns


class PathsDataset(data.Dataset):
    def __init__(self, paths, keys=None, filters={}):
        super(PathsDataset, self).__init__()

        if keys is not None:
            paths = [{k: path[k] for k in keys} for path in paths]

        paths_dict = collections.defaultdict(list)
        for path in paths:
            for k, v in path.items():
                paths_dict[k] = paths_dict[k] + list(v)
        self.paths_dict = paths_dict

        if len(self.paths_dict) == 0:
            self.length = 0
        else:
            k = list(self.paths_dict)[0]
            self.length = len(self.paths_dict[k])

        def default_func():
            return (lambda x: x)

        self.filters = collections.defaultdict(default_func)
        for k, v in filters.items():
            self.filters[k] = v

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {k: self.filters[k](self.paths_dict[k][idx]) for k in self.paths_dict}


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, collections.Iterable):
        return type(obj)((move_to_device(v, device) for v in obj))
    else:
        raise NotImplementedError
