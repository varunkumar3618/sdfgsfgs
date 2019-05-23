import numpy as np


class Schedule(object):
    def value(self, step):
        raise NotADirectoryError


class ConstantSchedule(Schedule):
    def __init__(self, constant):
        super(ConstantSchedule, self).__init__()
        self.constant = constant

    def value(self, step):
        return self.constant


class LinearSchedule(Schedule):
    def __init__(self, initial_value, final_value, decay_steps):
        super(LinearSchedule, self).__init__()
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_steps = decay_steps

    def value(self, step):
        delta = (self.final_value - self.initial_value) / float(self.decay_steps)
        current_value = self.initial_value + delta * step
        if self.initial_value < self.final_value:
            current_value = np.clip(current_value, self.initial_value, self.final_value)
        else:
            current_value = np.clip(current_value, self.final_value, self.initial_value)
        return current_value

    def __str__(self):
        return 'LinearSchedule{{initial:{}, final:{}, steps:{}}}'.format(
            self.initial_value,
            self.final_value,
            self.decay_steps
        )
