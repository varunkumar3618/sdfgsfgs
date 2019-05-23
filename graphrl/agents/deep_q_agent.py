import numpy as np
import torch
import torch.nn.functional as F
from graphrl.agents.replay_buffer import ReplayBuffer

from graphrl.agents.agent import Agent, AgentParams
from graphrl.agents.optimizer import OptimizerParams
from graphrl.agents.schedule import LinearSchedule


class DeepQAgentParams(AgentParams):
    def __init__(self):
        super(DeepQAgentParams, self).__init__()
        self.agent_class = DeepQNetwork

        self.batch_size = 32
        self.gamma = 0.99

        self.double_dqn = False
        self.use_huber_loss = False
        self.heatup_steps = 0
        self.update_freq_steps = 1000
        self.update_target_weights_freq_steps = 10000
        self.replay_buffer_size = 1000000

        self.train_epsilon_schedule = LinearSchedule(1, 0.1, 1000000)
        self.test_epsilon = 0.05

        self.optimizer_params = OptimizerParams()

        self.online_q_net = None
        self.target_q_net = None


class DeepQNetwork(Agent):
    def __init__(self, params):

        online_q_net = params.online_q_net.to(params.device)
        target_q_net = params.target_q_net.to(params.device)

        nets = {
            'online_q_net': online_q_net,
            'target_q_net': target_q_net
        }

        super(DeepQNetwork, self).__init__(params=params, nets=nets)

        self.online_q_net = online_q_net
        self.target_q_net = target_q_net
        self.optimizer = self.params.optimizer_params.make_optimizer(self.online_q_net)

        self.replay_buffer = ReplayBuffer(self.params.replay_buffer_size)

        # Store an obs until we see the effect
        self.obs = None

        self.last_update_step = 0
        self.last_update_target_weights_step = 0

    def train_on_env_reset(self, obs):
        self.obs = obs

    def train_on_env_step(self, obs, action, reward, done, info):
        self.replay_buffer.add(self.obs, action, reward, obs, done)

        if done:
            self.obs = None
        else:
            self.obs = obs

        self.maybe_update()

    def train_on_env_abort(self):
        self.obs = None

    def act(self, obs, training):
        if self.mode == 'train' and self.CTR_TRAIN_STEPS < self.params.heatup_steps:
            return self.train_env.action_space.sample()

        self.online_q_net.eval()

        obs = self.filter_obs([obs])
        obs = obs.to(self.params.device)
        with torch.no_grad():
            qs = self.online_q_net(obs)[0]
            best_action = int(qs.argmax())

        if training:
            epsilon = self.params.train_epsilon_schedule.value(self.CTR_TRAIN_STEPS - self.params.heatup_steps)
            self.log_scalar('train.epsilon.bystep', epsilon, self.CTR_TRAIN_STEPS)
        else:
            epsilon = self.params.test_epsilon

        if np.random.rand() > epsilon:
            action = best_action
        else:
            action = self.train_env.action_space.sample()

        return action

    def maybe_update(self):
        if self.CTR_TRAIN_STEPS < self.params.heatup_steps:
            return

        if self.CTR_TRAIN_STEPS - self.last_update_target_weights_step >= self.params.update_target_weights_freq_steps:
            self.target_q_net.load_state_dict(self.online_q_net.state_dict())
            self.last_update_target_weights_step = self.CTR_TRAIN_STEPS

        if self.CTR_TRAIN_STEPS - self.last_update_step >= self.params.update_freq_steps:
            self.update()
            self.last_update_step = self.CTR_TRAIN_STEPS

    def update(self):
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(self.params.batch_size)

        obs = self.filter_obs(obs).to(self.params.device)
        next_obs = self.filter_obs(next_obs).to(self.params.device)

        rewards, dones = [np.array(x, dtype=np.float32) for x in [rewards, dones]]
        actions = np.array(actions)
        actions, rewards, dones = [torch.from_numpy(x).to(self.params.device) for x in [actions, rewards, dones]]

        self.online_q_net.train()
        self.target_q_net.eval()

        self.optimizer.zero_grad()

        qs_from_online = self.online_q_net(obs)
        value_preds = qs_from_online.gather(1, actions[:, None])[:, 0]

        next_qs_from_target = self.target_q_net(next_obs).detach()

        if self.params.double_dqn:
            next_qs_from_online = self.online_q_net(next_obs).detach()
            next_actions = next_qs_from_online.argmax(1)
        else:
            next_actions = next_qs_from_target.argmax(1)

        next_values = next_qs_from_target.gather(1, next_actions[:, None])[:, 0]
        value_targets = rewards + (1 - dones) * self.params.gamma * next_values

        if self.params.use_huber_loss:
            loss = F.smooth_l1_loss(value_preds, value_targets)
        else:
            loss = F.mse_loss(value_preds, value_targets)

        loss.backward()
        self.optimizer.step()

        loss = float(loss)
        self.log_scalar('train.loss.bystep', loss, self.CTR_TRAIN_STEPS)
