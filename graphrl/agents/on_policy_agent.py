from graphrl.agents.agent import Agent, AgentParams


class OnPolicyAgentParams(AgentParams):
    def __init__(self):
        super(OnPolicyAgentParams, self).__init__()
        self.clear_during_episode = True

        self.update_freq_counter = 'episodes'
        self.update_freq_episodes = 5
        self.update_freq_steps = 1000


class OnPolicyAgent(Agent):
    def __init__(self, params):
        super(OnPolicyAgent, self).__init__(params=params)

        # Collect paths here as training progresses
        self.paths = []
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.last_update_step = 0
        self.last_update_episode = 0

    def train_on_env_reset(self, obs):
        self.observations.append(obs)

    def pop_current_path(self, should_save_last_obs):
        observations = self.observations
        actions = self.actions
        rewards = self.rewards
        dones = self.dones

        if should_save_last_obs:
            self.observations = [observations[-1]]
        else:
            self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

        if len(actions) > 0:
            path = {
                'observation': observations[:-1],
                'next_observation': observations[1:],
                'action': actions,
                'reward': rewards,
                'done': dones
            }
            self.paths.append(path)

    def update_model(self):
        self.last_update_step = self.CTR_TRAIN_STEPS
        self.last_update_episode = self.CTR_TRAIN_EPISODES

        paths = self.paths
        self.paths = []

        self.train_on_paths(paths)

    def maybe_clear_and_act(self, done, aborted):
        should_update = False
        if self.params.update_freq_counter == 'episodes':
            if self.CTR_TRAIN_EPISODES - self.last_update_episode >= self.params.update_freq_episodes:
                should_update = True
        elif self.params.update_freq_counter == 'steps':
            if self.CTR_TRAIN_STEPS - self.last_update_step >= self.params.update_freq_steps:
                should_update = True
        else:
            raise ValueError('Unknown update counter {}'.format(self.params.update_freq_counter))

        if done or aborted:
            self.pop_current_path(should_save_last_obs=False)
            if should_update:
                self.update_model()
        elif self.params.clear_during_episode and should_update:
            self.pop_current_path(should_save_last_obs=True)
            self.update_model()

    def train_on_env_step(self, obs, action, reward, done, info):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        self.maybe_clear_and_act(done=done, aborted=False)

    def train_on_env_abort(self):
        self.maybe_clear_and_act(done=False, aborted=True)
