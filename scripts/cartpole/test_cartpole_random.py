import gym
import sacred

from graphrl.agents.random_agent import RandomAgentParams


ex = sacred.Experiment('test_cartpole_random')


@ex.automain
def main(_run):
    env = gym.make('CartPole-v0')
    agent_params = RandomAgentParams()
    agent_params.sacred_run = _run
    agent_params.env = env
    agent_params.mode = 'test'

    agent = agent_params.make_agent()
    agent.run()
