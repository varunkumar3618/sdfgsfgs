import torch
import os
import numbers
import time
import numpy as np
import tempfile

from graphrl.agents.filters import RewardFilter, ObservationFilter
from graphrl.agents.stopping import StoppingCondition


class AgentParams(object):
    def __init__(self):
        super(AgentParams, self).__init__()

        self.mode = 'train'

        # Stopping conditions
        self.min_train_steps = 10000000000
        self.use_min_train_steps = True

        self.min_train_episodes = 100000
        self.use_min_train_episodes = False

        self.no_progress_steps = 50000
        self.use_no_progress_steps = False

        # Train schedule params
        self.eval_freq_episodes = 50
        self.max_steps_per_train_episode = None

        # Eval schedule params
        self.eval_episodes = 10

        # Test schedule params
        self.test_episodes = 1000
        self.max_steps_per_test_episode = None

        # Other params
        self.device_name = 'cuda'
        self.device = None

        self.print_actions = False

        # Filters
        # NOTE: the Agent class takes care of applying the reward filter but not the observation filter.
        # This is to allow memory-expensive processing such as stacking frames to be done as late as possible.
        # Subclasses should call self.filter_obs to use the observation filter.
        self.reward_filter = RewardFilter()
        self.obs_filter = ObservationFilter()

        self.agent_class = Agent

        self.sacred_run = None

        self.train_env = None
        self.test_envs = []

        self.should_load_nets = False
        self.load_nets_folder = './model'

        self.save_freq_episodes = 500

        self.tensorboard_logdir = './log'
        self.use_tensorboard = False

        self.custom_stopping_cond = StoppingCondition()

    @property
    def env(self):
        return self.train_env

    @env.setter
    def env(self, value):
        self.train_env = value
        self.test_envs.append(value)

    def make_agent(self):
        print('Agent params class {}.'.format(type(self)))
        print('------------------------------------------')

        # Print string params
        for k, v in vars(self).items():
            if isinstance(v, str):
                print("{} : {}".format(k, v))

        # Print number params
        for k, v in vars(self).items():
            if isinstance(v, numbers.Number):
                print("{} : {}".format(k, v))

        # Print other params
        for k, v in vars(self).items():
            if not isinstance(v, str) and not isinstance(v, numbers.Number):
                print("{} : {}".format(k, v))

        self.device = torch.device(self.device_name)
        return self.agent_class(params=self)


class Agent(object):
    def __init__(self, params, nets=None):
        super(Agent, self).__init__()
        self.params = params
        self.mode = self.params.mode
        self.train_env = self.params.train_env
        self.test_envs = self.params.test_envs
        self.sacred_run = self.params.sacred_run
        self.device = self.params.device

        self.custom_stopping_cond = self.params.custom_stopping_cond

        # this counter is incremented when an action is taken.
        self.CTR_TRAIN_STEPS = 0
        # this counter is incremented when an episode ends or is aborted
        self.CTR_TRAIN_EPISODES = 0

        self.best_eval_results = None
        self.best_eval_steps = 0

        self.last_eval_episode = 0
        self.last_save_episode = 0

        if nets is None:
            nets = {}
        self.nets = nets

        print('Nets')
        for name, net in self.nets.items():
            print('Net {} has {} parameters.'.format(name, sum(p.numel() for p in net.parameters())))

        self.load_nets()

        self.open_files = {}

        if self.params.use_tensorboard:
            import tensorboardX
            self.summary_writer = tensorboardX.SummaryWriter(log_dir=self.params.tensorboard_logdir)

    # Subclasses should implement these methods

    def act(self, obs, training):
        raise NotImplementedError

    def train_on_env_reset(self, obs):
        pass

    def train_on_env_step(self, obs, action, reward, done, info):
        pass

    def train_on_env_abort(self):
        pass

    # Methods below this are implemented here

    def filter_obs(self, obs_list):
        return self.params.obs_filter(obs_list)

    def run_episode(self, env, training):
        rewards = []
        episode_steps = 0
        done = False

        start_time = time.time()

        obs = env.reset()

        if training:
            self.train_on_env_reset(obs)

        while True:
            action = self.act(obs, training=training)
            if self.params.print_actions:
                print('Action: {}'.format(action))
            obs, reward, done, info = env.step(action)

            if training:
                self.CTR_TRAIN_STEPS += 1
                if done:
                    self.CTR_TRAIN_EPISODES += 1

                filtered_reward = self.params.reward_filter(reward)
                self.train_on_env_step(obs, action, filtered_reward, done, info)

            rewards.append(reward)
            episode_steps += 1

            if done:
                break

            if training:
                if self.params.max_steps_per_train_episode is not None and episode_steps >= self.params.max_steps_per_train_episode:
                    self.CTR_TRAIN_EPISODES += 1
                    self.train_on_env_abort()  # in case cleanup is required
                    break
            else:
                if self.params.max_steps_per_test_episode is not None and episode_steps >= self.params.max_steps_per_test_episode:
                    break

        end_time = time.time()
        episode_speed = episode_steps / float(end_time - start_time)

        episode_reward = sum(rewards)
        episode_completed = int(done)

        self.custom_stopping_cond.report_train_episode(episode_reward)

        return episode_reward, episode_steps, episode_completed, episode_speed

    def log_scalar(self, name, value, step):
        self.sacred_run.log_scalar(name, value, step)
        if self.params.use_tensorboard:
            tensorboard_name = name.replace('.', '/')
            self.summary_writer.add_scalar(tensorboard_name, value, step)

    def log_episode_to_sacred(self, episode_reward, episode_steps, episode_completed, episode_speed, phase, suffix, index):
        self.log_scalar('{}.episode_reward.{}'.format(phase, suffix), episode_reward, index)
        self.log_scalar('{}.episode_steps.{}'.format(phase, suffix), episode_steps, index)
        self.log_scalar('{}.episode_completed.{}'.format(phase, suffix), episode_completed, index)
        self.log_scalar('{}.episode_speed.{}'.format(phase, suffix), episode_speed, index)

    def log_train_to_sacred(self, name, value):
        self.log_scalar('train.{}.bystep'.format(name), value, self.CTR_TRAIN_STEPS)
        self.log_scalar('train.{}.byepisode'.format(name), value, self.CTR_TRAIN_EPISODES)

    def evaluate_on(self, env, name):
        episode_reward_list = []
        episode_steps_list = []
        episode_completed_list = []
        episode_speed_list = []

        for i in range(self.params.eval_episodes):
            episode_reward, episode_steps, episode_completed, episode_speed = self.run_episode(env, training=False)
            episode_reward_list.append(episode_reward)
            episode_steps_list.append(episode_steps)
            episode_completed_list.append(episode_completed)
            episode_speed_list.append(episode_speed)

        episode_reward_mean, episode_reward_std = np.mean(episode_reward_list), np.std(episode_reward_list)
        episode_steps_mean, episode_steps_std = np.mean(episode_steps_list), np.std(episode_steps_list)
        episode_completed_mean, episode_completed_std = np.mean(episode_completed_list), np.std(episode_completed_list)
        episode_speed_mean, episode_speed_std = np.mean(episode_speed_list), np.std(episode_speed_list)

        for suffix, index in [('bystep', self.CTR_TRAIN_STEPS), ('byepisode', self.CTR_TRAIN_EPISODES)]:
            self.log_scalar('{}.episode_reward_mean.{}'.format(name, suffix), episode_reward_mean, index)
            self.log_scalar('{}.episode_reward_std.{}'.format(name, suffix), episode_reward_std, index)
            self.log_scalar('{}.episode_steps_mean.{}'.format(name, suffix), episode_steps_mean, index)
            self.log_scalar('{}.episode_steps_std.{}'.format(name, suffix), episode_steps_std, index)
            self.log_scalar('{}.episode_completed_mean.{}'.format(name, suffix), episode_completed_mean, index)
            self.log_scalar('{}.episode_completed_std.{}'.format(name, suffix), episode_completed_std, index)
            self.log_scalar('{}.episode_speed_mean.{}'.format(name, suffix), episode_speed_mean, index)
            self.log_scalar('{}.episode_speed_std.{}'.format(name, suffix), episode_speed_std, index)

        print('Step {}. Evaluation episode on {}. Reward mean {:.2f}, reward std {:.2f}, average steps: {}, completed: {}, speed: {:.2f} f/s'
              .format(self.CTR_TRAIN_STEPS, name, episode_reward_mean, episode_reward_std, episode_steps_mean, episode_completed_mean, episode_speed_mean))

        return episode_reward_mean

    def evaluate(self):
        self.evaluate_on(self.train_env, 'eval_train')

        results = []
        for i, env in enumerate(self.test_envs):
            mean_reward = self.evaluate_on(env, 'eval_test_{}'.format(i + 1))
            self.custom_stopping_cond.report_eval_episode_mean(i, mean_reward)
            results.append(mean_reward)

        if self.best_eval_results is None or all(cur_result > prev_result for cur_result, prev_result in zip(results, self.best_eval_results)):
            self.best_eval_results = results
            self.best_eval_steps = self.CTR_TRAIN_STEPS
            self.save_nets(is_best=True)
        self.last_eval_episode = self.CTR_TRAIN_EPISODES

    def test(self):
        for i, env in enumerate(self.test_envs):
            print('Testing on test env {}'.format(i))
            for j in range(self.params.test_episodes):
                episode_reward, episode_steps, episode_completed, episode_speed = self.run_episode(env, training=False)
                self.log_episode_to_sacred(episode_reward, episode_steps, episode_completed, episode_speed, 'test_{}'.format(i), 'byepisode', j)

                print('Test episode no {}. Reward: {:.2f}, steps: {}, completed: {}, speed: {:.2f} f/s'
                      .format(j, episode_reward, episode_steps, episode_completed, episode_speed))

    def load_nets(self):
        if not self.params.should_load_nets:
            return
        for name, net in self.nets.items():
            net.load_state_dict(torch.load(os.path.join(self.params.load_nets_folder, '{}.pth'.format(name))))

    def save_nets(self, is_best=False):
        for name, net in self.nets.items():
            new_file, filename = tempfile.mkstemp()
            torch.save(net.state_dict(), filename)
            self.sacred_run.add_artifact(filename, name='{}_{}.pth'.format(name, self.CTR_TRAIN_EPISODES))

            if is_best:
                print('Saving best model at step {}'.format(self.CTR_TRAIN_STEPS))
                self.sacred_run.add_artifact(filename, name='{}_best.pth'.format(name))

        print('Saving at step {}'.format(self.CTR_TRAIN_STEPS))
        self.last_save_episode = self.CTR_TRAIN_EPISODES

    def check_min_train_steps(self):
        if self.params.use_min_train_steps:
            return self.CTR_TRAIN_STEPS > self.params.min_train_steps
        return True

    def check_min_train_episodes(self):
        if self.params.use_min_train_episodes:
            return self.CTR_TRAIN_EPISODES > self.params.min_train_episodes
        return True

    def check_no_progress_steps(self):
        if self.params.use_no_progress_steps:
            return self.CTR_TRAIN_STEPS - self.best_eval_steps > self.params.no_progress_steps
        return True

    def train(self):

        while True:

            # Check stopping conditions
            if self.check_min_train_steps() and self.check_min_train_episodes() and self.check_no_progress_steps():
                break

            if self.custom_stopping_cond.should_stop(self):
                break

            episode_reward, episode_steps, episode_completed, episode_speed = self.run_episode(self.train_env, training=True)
            self.log_episode_to_sacred(episode_reward, episode_steps, episode_completed, episode_speed, 'train', 'bystep', self.CTR_TRAIN_STEPS)
            self.log_episode_to_sacred(episode_reward, episode_steps, episode_completed, episode_speed, 'train', 'byepisode', self.CTR_TRAIN_EPISODES)
            print('Step {}. Train episode no {}. Reward: {:.2f}, steps: {}, completed: {}, speed: {:.2f} f/s'
                  .format(self.CTR_TRAIN_STEPS, self.CTR_TRAIN_EPISODES, episode_reward, episode_steps, episode_completed, episode_speed))

            if self.CTR_TRAIN_EPISODES - self.last_eval_episode >= self.params.eval_freq_episodes:
                self.evaluate()

            if self.CTR_TRAIN_EPISODES - self.last_save_episode >= self.params.save_freq_episodes:
                self.save_nets()

        self.evaluate()
        self.save_nets()

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()
        else:
            raise ValueError('Agent mode must be train or test.')
