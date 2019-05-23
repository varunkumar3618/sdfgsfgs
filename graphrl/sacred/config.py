import os
from sacred.observers import MongoObserver, SlackObserver
from sacred.commandline_options import CommandLineOption
from visdom_observer.visdom_observer import VisdomObserver


class MongoOption(CommandLineOption):
    '''Add a mongo db observer.'''

    @classmethod
    def apply(cls, args, run):
        url = os.environ['SACRED_MONGO_URL']
        db_name = os.environ['SACRED_MONGO_DB_NAME']
        mongo = MongoObserver.create(url=url, db_name=db_name)
        run.observers.append(mongo)


class VisdomOption(CommandLineOption):
    '''Add a visdom observer.'''

    @classmethod
    def apply(cls, args, run):
        run.observers.append(VisdomObserver())


def add_params(params, prefix, _config):
    d = _config[prefix]
    for k, v in d.items():
        if v is None:
            continue
        if not hasattr(params, k):
            raise ValueError('Unknown parameter {}'.format(k))
        else:
            setattr(params, k, v)


def maybe_add_slack(ex):
    if os.path.exists('slack.json'):
        slack_obs = SlackObserver.from_config('slack.json')
        ex.observers.append(slack_obs)
        print('Added slack observer.')
