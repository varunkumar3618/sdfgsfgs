
from graphrl.agents.agent import Agent, AgentParams


class RandomAgentParams(AgentParams):
    def __init__(self):
        super(RandomAgentParams, self).__init__()
        self.agent_class = RandomAgent


class RandomAgent(Agent):
    def train_iteration(self):
        pass

    def act(self, obs, training):
        return self.params.env.action_space.sample()
