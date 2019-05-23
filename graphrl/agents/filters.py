import numpy as np
from torch.utils.data.dataloader import default_collate


class RewardFilter(object):
    def __call__(self, value):
        return value

    def __str__(self):
        return 'RewardFilter()'


class RewardRescaleFilter(RewardFilter):
    def __init__(self, scale):
        super(RewardRescaleFilter, self).__init__()
        self.scale = scale

    def __call__(self, value):
        return value / self.scale

    def __str__(self):
        return 'RewardRescaleFilter({})'.format(self.scale)


class ObservationFilter(object):
    def __call__(self, values):
        return default_collate(values)

    def output_space(self, input_space):
        return input_space

    def __str__(self):
        return 'ObservationFilter()'


class AtariObservationFilter(ObservationFilter):
    def __call__(self, values):
        values = [np.array(value) for value in values]
        values = default_collate(values)
        return values

    def __str__(self):
        return 'AtariObservationFilter()'.format()
