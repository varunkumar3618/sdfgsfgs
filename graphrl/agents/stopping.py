

class StoppingCondition(object):
    def report_train_episode(self, episode_reward):
        pass

    def report_eval_episode_mean(self, env_idx, episode_reward):
        pass

    def should_stop(self, agent):
        return False


class DoesntTrainStoppingCondition(StoppingCondition):
    def __init__(self, min_episodes, doesnt_train_episodes, bad_reward):
        super(DoesntTrainStoppingCondition, self).__init__()
        self.min_episodes = min_episodes
        self.doesnt_train_episodes = doesnt_train_episodes
        self.bad_reward = bad_reward

        self.train_rewards = []

    def report_train_episode(self, episode_reward):
        self.train_rewards.append(episode_reward)

    def should_stop(self, agent):
        if agent.CTR_TRAIN_EPISODES < self.min_episodes:
            return False
        recent_rewards = self.train_rewards[-self.doesnt_train_episodes:]
        if len(recent_rewards) < self.doesnt_train_episodes:
            return False
        should_stop = True
        for reward in recent_rewards:
            if reward > self.bad_reward:
                should_stop = False
        return should_stop
