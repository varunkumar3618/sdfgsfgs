import sacred

from graphrl.agents.random_agent import RandomAgentParams
from graphrl.environments.pacman.pacman_gym import PacmanEnv
from graphrl.environments.wrappers import RenderEnv


ex = sacred.Experiment('test_pacman_random')


@ex.config
def config():
    env = {
        'layout_file': 'assets/pacman/mediumClassic',
        'ghost_type': 'random',
        'render': True
    }


@ex.capture
def build_env(env):
    render = env.pop('render')
    env = PacmanEnv(**env)
    if render:
        env = RenderEnv(env)
    return env


@ex.automain
def main(_run):
    env = build_env()
    agent_params = RandomAgentParams()
    agent_params.sacred_run = _run
    agent_params.env = env
    agent_params.mode = 'test'

    agent = agent_params.make_agent()
    agent.run()
