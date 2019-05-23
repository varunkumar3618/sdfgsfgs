import gym
import gym.spaces
import numpy as np
import networkx as nx

from graphrl.environments.pacman.pacman import ClassicGameRules, PacmanRules
from graphrl.environments.pacman.ghostAgents import RandomGhost, DirectionalGhost
from graphrl.environments.pacman.game import Agent, Directions
import graphrl.environments.pacman.textDisplay as textDisplay
import graphrl.environments.pacman.layout as layout
from graphrl.environments.pycolab_wrappers import OneHotEnv
from graphrl.environments.pacman.util import nearestPoint


class MyPacman(Agent):
    def __init__(self, pacman_env):
        super(MyPacman, self).__init__()
        self.pacman_env = pacman_env
        self.action = None

    def getAction(self, state):
        return self.action


class PacmanEnv(gym.Env):
    def __init__(self, layout_file, ghost_type):
        super(PacmanEnv, self).__init__()
        self.ghost_type = ghost_type
        self.layout_file = layout_file

        if ghost_type == 'random':
            ghost_cls = RandomGhost
        elif ghost_type == 'directional':
            ghost_cls = DirectionalGhost
        else:
            raise ValueError('Unrecognized ghost type: {}.'.format(ghost_type))
        self.ghost_cls = ghost_cls

        self.game_rules = ClassicGameRules()

        self.display = textDisplay.NullGraphics()
        self.layout = layout.getLayout(self.layout_file)
        self.num_ghosts = self.layout.getNumGhosts()

        self.pacman_agent = MyPacman(self)
        self.ghost_agents = [self.ghost_cls(i + 1) for i in range(self.num_ghosts)]
        self.agents = [self.pacman_agent] + self.ghost_agents

        self.height, self.width = self.layout.height, self.layout.width

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, dtype=np.uint8, shape=(self.height, self.width))

        self.game = None
        self.score = None

        self.last_state = None

    def convert_state(self, state):
        ghost_states = state.getGhostStates()
        scared_ghost_positions = [ghost_state.getPosition() for ghost_state in ghost_states if ghost_state.scaredTimer > 0]

        state = state.data.map_str()

        for c in ['>', '<', '^', 'v']:
            state = state.replace(c, 'P')

        state = state.split('\n')
        state = [list(line) for line in state]
        state = list(reversed(state))

        for point in scared_ghost_positions:
            x, y = nearestPoint(point)

            if state[y][x] == 'G' or state[y][x] == 'H':
                state[y][x] = 'H'

        new_state = np.zeros((self.height, self.width), dtype=np.uint8)

        for i in range(self.height):
            for j in range(self.width):
                new_state[i, j] = ord(state[i][j])

        return new_state

    def convert_action(self, action):
        if action == 0:
            action = Directions.NORTH
        elif action == 1:
            action = Directions.SOUTH
        elif action == 2:
            action = Directions.WEST
        elif action == 3:
            action = Directions.EAST
        else:
            raise ValueError('Invalid action {}.'.format(action))

        if action in PacmanRules.getLegalActions(self.game.state):
            return action
        else:
            return None

    def reset(self):
        self.game = self.game_rules.newGame(self.layout, self.pacman_agent, self.ghost_agents, self.display)
        self.score = self.game.state.getScore()

        for i in range(len(self.agents)):
            agent = self.agents[i]
            if ("registerInitialState" in dir(agent)):
                agent.registerInitialState(self.game.state.deepCopy())
        state = self.convert_state(self.game.state)
        self.last_state = state
        return state

    def step(self, action):
        self.pacman_agent.action = self.convert_action(action)

        for i, agent in enumerate(self.agents):
            if self.game.state.isWin() or self.game.state.isLose():
                break
            action = agent.getAction(self.game.state.deepCopy())
            if action is not None:
                self.game.state = self.game.state.generateSuccessor(i, action)

        done = self.game.state.isWin() or self.game.state.isLose()
        state = self.convert_state(self.game.state)

        prev_score = self.score
        self.score = self.game.state.getScore()
        reward = self.score - prev_score

        self.last_state = state
        return state, reward, done, {}

    def render(self, mode='human'):
        if mode != 'human':
            raise ValueError('Invalid mode: {}'.format(mode))

        state = '\n'.join([''.join([chr(x) for x in line]) for line in self.last_state])

        print(state)
        print('Score: {}'.format(self.score))


def make_pacman_env(layout_file, ghost_type, encode_onehot=False):
    env = PacmanEnv(layout_file, ghost_type)
    if encode_onehot:
        characters = ['%', ' ', '.', 'G', 'H', 'o', 'P']
        values = [ord(c) for c in characters]
        env = OneHotEnv(env, values)
    return env


def make_pacman_knowledge_graph(ghost_type, no_edges=False, fully_connected=False, fully_connected_distinct=False, same_edge_feats=False):
    G = nx.DiGraph()

    def get_feature(num_features, idx, same_edge_feats=False):
        if same_edge_feats:
            idx = 0

        feature = np.zeros((num_features,), dtype=np.float32)
        feature[idx] = 1
        return feature

    characters = ['%', ' ', '.', 'G', 'H', 'o', 'P']

    for i, c in enumerate(characters):
        G.add_node(c, node_feature=get_feature(len(characters), i))

    num_edge_features = 3

    if no_edges:
        pass
    elif fully_connected:
        num_edge_features = 3
        feature = np.array([1, 0, 0], dtype=np.float32)

        for n1 in G.nodes:
            for n2 in G.nodes:
                if n1 != n2:
                    G.add_edge(n1, n2, edge_feature=feature)
    elif fully_connected_distinct:
        num_edge_features = len(G.nodes) * (len(G.nodes) - 1)

        for i, n1 in enumerate(G.nodes):
            for j, n2 in enumerate(G.nodes):
                if n1 != n2:
                    idx = i * (len(G.nodes) - 1) + j
                    G.add_edge(n1, n2, edge_feature=get_feature(num_edge_features, idx))
    else:
        num_edge_features = 6

        G.add_edge('P', '%', edge_feature=get_feature(num_edge_features, 0), same_edge_feats=same_edge_feats)
        G.add_edge('G', '%', edge_feature=get_feature(num_edge_features, 0), same_edge_feats=same_edge_feats)
        G.add_edge('H', '%', edge_feature=get_feature(num_edge_features, 0), same_edge_feats=same_edge_feats)

        G.add_edge('P', '.', edge_feature=get_feature(num_edge_features, 1), same_edge_feats=same_edge_feats)

        G.add_edge('P', 'o', edge_feature=get_feature(num_edge_features, 2), same_edge_feats=same_edge_feats)

        G.add_edge('G', 'P', edge_feature=get_feature(num_edge_features, 3), same_edge_feats=same_edge_feats)

        G.add_edge('P', 'H', edge_feature=get_feature(num_edge_features, 4), same_edge_feats=same_edge_feats)

        G.add_edge('G', 'H', edge_feature=get_feature(num_edge_features, 5), same_edge_feats=same_edge_feats)

    G = nx.relabel_nodes(G, {entity: ord(entity) for entity in G.nodes})
    return [ord(char) for char in characters], G, len(characters), num_edge_features
