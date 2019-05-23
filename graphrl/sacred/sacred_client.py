import pymongo
import os
import tabulate
import dateutil
import numpy as np
import matplotlib.pyplot as plt


def flatten_dict(d):
    ret = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
            for subk, subv in v.items():
                ret['{}.{}'.format(k, subk)] = subv
        else:
            ret[k] = '{}'.format(v)
    return ret


def bucket(steps, values, ends, lower=True):
    result = []
    idx = 0

    for end in ends:
        end_result = []
        while True:
            if idx >= len(steps):
                break
            step, value = steps[idx], values[idx]

            if (lower and step <= end) or (step < end):
                end_result.append(value)
                idx += 1
            else:
                break
        result.append(end_result)

    return result


def bucket_reduce(steps, values, ends, lower=True, func=np.mean):
    result = bucket(steps, values, ends, lower=lower)
    result = [(i, func(val)) for i, val in enumerate(result) if len(val) > 0]
    result_steps, result_values = [v[0] for v in result], [v[1] for v in result]
    return result_steps, result_values


def get_run_times(run):
    time = run['start_time']
    from_zone = dateutil.tz.tzutc()
    to_zone = dateutil.tz.tzlocal()
    local_time = time.replace(tzinfo=from_zone).astimezone(to_zone)

    date_str = local_time.date().strftime('%Y-%m-%d')
    time_str = local_time.time().strftime('%H:%M:%S')
    return date_str, time_str


def smooth_series(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    y = y[(window_len // 2):-(window_len // 2)]
    return y


class SacredClient(object):
    def __init__(self):
        super(SacredClient, self).__init__()
        hostname = os.environ['SACRED_MONGO_HOST']
        port = int(os.environ['SACRED_MONGO_PORT'])
        database_name = os.environ['SACRED_MONGO_DB_NAME']
        self.client = pymongo.MongoClient(hostname, port)
        self.database = self.client[database_name]

        self.runs = self.database['runs']
        self.metrics = self.database['metrics']

    def get_run_list(self, filter_dict={}):
        run_list = []
        for run in self.runs.find(filter_dict):
            run_list.append(run)
        return run_list

    def get_run_by_id(self, run_id):
        result = list(self.runs.find({'_id': run_id}))
        if len(result) > 1:
            raise ValueError('Duplicates.')
        return result[0]

    def get_metrics_by_id(self, run_id):
        '''
        Return a dict keyed by the name of the metric.
        Each metric is a pair (steps, values)
        '''
        result = list(self.metrics.find({'run_id': run_id}))

        result_dict = {}
        for metric in result:
            result_dict[metric['name']] = (metric['steps'], metric['values'])
        return result_dict

    def get_metric_names(self, run_id):
        result = set()
        for metric in self.metrics.find({'run_id': run_id}):
            result.add(metric['name'])
        return list(result)

    def get_metric(self, run_id, name):
        result = list(self.metrics.find({'run_id': run_id, 'name': name}))
        if len(result) > 1:
            raise ValueError('Duplicates.')
        result = result[0]
        return result['steps'], result['values']

    def extract_configs_from_run_list(self, run_list):
        configs = {}
        keys = set()

        for run in run_list:
            run_id = run['_id']
            config = run['config']
            config = flatten_dict(config)
            configs[run_id] = config
            keys.update(config.keys())
        return configs, list(keys)

    def display_metric(self, run_id, name, xlabel='steps', smooth=False, smoothing_params={}):
        steps, values = self.get_metric(run_id, name)
        if smooth:
            values = smooth_series(np.array(values), **smoothing_params)
        fig, ax = plt.subplots()
        ax.plot(steps, values)
        plt.xlabel(xlabel)
        plt.ylabel(name)
        plt.show()

    def display_metric_trans(self, run_id, name, ends_name, lower=True, xlabel='episodes', func=np.mean):
        ends = self.get_metric(run_id, ends_name)[0]
        steps, values = self.get_metric(run_id, name)
        episodes, values = bucket_reduce(steps, values, ends, lower=lower, func=func)

        fig, ax = plt.subplots()
        ax.plot(episodes, values)
        plt.xlabel(xlabel)
        plt.ylabel(name)
        plt.show()

    def display(self, filter_dict={}):
        from IPython.display import display, HTML
        table = self.runs_table(filter_dict=filter_dict)
        display(HTML(table))

    def display_output(self, run_id):
        run = self.get_run_by_id(run_id)
        print(run['captured_out'])

    def runs_table(self, tablefmt='html', filter_dict={}):
        run_list = self.get_run_list(filter_dict=filter_dict)
        configs, config_keys = self.extract_configs_from_run_list(run_list)

        headers = ['ID', 'Experiment Name', 'Start Date', 'Start Time'] + config_keys
        table = []
        for i, run in enumerate(run_list):
            row = []
            run_id = run['_id']
            row.append(run_id)
            row.append(run['experiment']['name'])

            start_date, start_time = get_run_times(run)
            row.append(start_date)
            row.append(start_time)

            config = configs[run_id]
            for k in config_keys:
                if k in config:
                    row.append(config[k])
                else:
                    row.append('')
            table.append(row)
        return tabulate.tabulate(table, headers=headers, tablefmt=tablefmt)
