import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
import torch.utils.data as data
import collections

from graphrl.agents.on_policy_agent import OnPolicyAgent, OnPolicyAgentParams
from graphrl.agents.optimizer import OptimizerParams
from graphrl.agents.utils import PathsDataset, move_to_device, compute_returns, make_collate_fn


class ActorCriticAgentParams(OnPolicyAgentParams):
    def __init__(self):
        super(ActorCriticAgentParams, self).__init__()
        self.batch_size = 32

        # the policy and value advantage methods are both advantage estimation methods.
        # they need not be the same.
        self.policy_advantage_method = 'gae'
        self.value_advantage_method = 'return'
        self.gamma = 0.99
        self.gae_lambda = 0.96
        self.use_huber_loss = False
        self.policy_loss_weight = 1.
        self.value_loss_weight = 0.5
        self.policy_entropy_weight = 0.
        self.optimizer_params = OptimizerParams()
        self.policy_value_net = None
        self.agent_class = ActorCriticAgent
        self.clear_during_episode = True


def td_residuals(values, next_values, dones, rewards, gamma):
    if dones[-1]:
        last_next_value = 0
    else:
        last_next_value = next_values[-1]

    next_values = np.concatenate([values[1:], [last_next_value]])
    return np.array(rewards) + gamma * next_values - values


def discount_and_sum(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class ActorCriticAgent(OnPolicyAgent):
    def __init__(self, params):
        super(ActorCriticAgent, self).__init__(params=params)
        '''
        policy_value_net.forward shoulf return a pair (policy, value)
        '''

        self.policy_value_net = self.params.policy_value_net.to(self.params.device)
        self.optimizer = self.params.optimizer_params.make_optimizer(self.policy_value_net)

    def act(self, obs, training):
        self.policy_value_net.eval()

        obs = self.filter_obs([obs]).to(self.params.device)
        with torch.no_grad():
            action_dist = self.policy_value_net(obs)[0]
        if training:
            action = action_dist.sample()[0]
        else:
            action = action_dist.logits.argmax(1)[0]
        return action.cpu().numpy()

    def add_values_to_paths(self, paths):
        # We want to compute value and next_value for each state without redoing work

        for path in paths:
            path['all_observation'] = list(path['observation']) + [path['next_observation'][-1]]

        self.policy_value_net.eval()

        val_dataset = PathsDataset(paths, keys=['all_observation'])
        val_dataloader = data.DataLoader(val_dataset,
                                         collate_fn=make_collate_fn({'all_observation': self.filter_obs}),
                                         batch_size=self.params.batch_size,
                                         shuffle=False,
                                         pin_memory=self.params.device.type == 'cuda')

        values = []
        for batch in val_dataloader:
            with torch.no_grad():
                value = self.policy_value_net(batch['all_observation'].to(self.params.device))[1]
                values.append(value.cpu().numpy())

        values = np.concatenate(values)

        path_lengths = [len(path['all_observation']) for path in paths]
        split_idx = 0
        split_idxs = []
        for path_length in path_lengths:
            split_idx += path_length
            split_idxs.append(split_idx)
        values = np.split(values, split_idxs)[:-1]
        for path, value in zip(paths, values):
            path['all_value'] = value
            path['value'] = value[:-1]
            path['next_value'] = value[1:]

    def make_advantages_and_targets(self, path, advantage_method):
        residuals = td_residuals(values=path['value'],
                                 next_values=path['next_value'],
                                 dones=path['done'],
                                 rewards=path['reward'],
                                 gamma=self.params.gamma)
        if advantage_method == 'bootstrap':
            advantages = residuals
            value_targets = np.array(path['value']) + advantages
        elif advantage_method == 'gae':
            advantages = discount_and_sum(residuals, self.params.gamma * self.params.gae_lambda)
            value_targets = np.array(path['value']) + advantages
        elif advantage_method == 'return':
            if path['done'][-1]:
                last_next_value = 0
            else:
                last_next_value = path['next_value'][-1]
            extended_rewards = list(path['reward']) + [last_next_value]
            extended_returns = compute_returns(extended_rewards, self.params.gamma, use_future_return=True)
            value_targets = np.array(extended_returns[:-1])
            advantages = value_targets - np.array(path['value'])
        else:
            raise ValueError('Unrecognized advantage method: {}'.format(self.params.advantage_method))
        return advantages, value_targets

    def augment_paths(self, paths):
        '''
        Adds 'advantage' and 'value_target' to paths.
        '''

        self.add_values_to_paths(paths)

        for path in paths:
            # Add advantage
            advantages, _ = self.make_advantages_and_targets(path, self.params.policy_advantage_method)
            path['advantage'] = advantages

            # Add value target
            _, value_targets = self.make_advantages_and_targets(path, self.params.value_advantage_method)
            path['value_target'] = value_targets

    def backward_on_batch(self, batch):
        batch = move_to_device(batch, self.params.device)

        obss = batch['observation']
        actions = batch['action']
        value_targets = batch['value_target']
        advantages = batch['advantage']

        action_dist, value = self.policy_value_net(obss)

        policy_loss = -torch.mean(action_dist.log_prob(actions) * advantages.float())
        policy_entropy = torch.mean(action_dist.entropy())

        if self.params.use_huber_loss:
            value_loss = F.smooth_l1_loss(value, value_targets.float())
        else:
            value_loss = F.mse_loss(value, value_targets.float())

        loss = self.params.policy_loss_weight * policy_loss
        loss = loss - self.params.policy_entropy_weight * policy_entropy
        loss = loss + self.params.value_loss_weight * value_loss

        loss.backward()

        loss, policy_loss, policy_entropy, value_loss = float(loss), float(policy_loss), float(policy_entropy), float(value_loss)
        bz = len(obss)

        losses = {'loss': loss, 'policy_loss': policy_loss, 'policy_entropy': policy_entropy, 'value_loss': value_loss}

        return bz, losses

    def train_on_paths(self, paths):
        self.augment_paths(paths)

        train_dataset = PathsDataset(paths,
                                     keys=['observation', 'action', 'value_target', 'advantage'])
        train_dataloader = data.DataLoader(train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           pin_memory=self.params.device.type == 'cuda',
                                           collate_fn=make_collate_fn({'observation': self.filter_obs}))

        losses_sums = collections.defaultdict(float)
        bz_sum = 0

        self.policy_value_net.train()
        self.optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            bz, losses = self.backward_on_batch(batch)

            bz_sum += bz
            for k, v in losses.items():
                losses_sums[k] += v * bz
        self.optimizer.step()

        for k, v in losses_sums.items():
            v = v / bz_sum
            self.log_train_to_sacred(k, v / bz_sum)
