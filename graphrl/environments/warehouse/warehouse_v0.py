import numpy as np
from pycolab import ascii_art, things
from pycolab.prefab_parts.sprites import MazeWalker
import networkx as nx

from graphrl.environments.pycolab_wrappers import load_art, OneHotEnv, PycolabMazeEnv
from graphrl.environments.wrappers import SampleEnv

# Characters in the pycolab game

BACKGROUND_CHARACTER = ' '

WALL_CHARACTER = '+'

PLAYER_CHARACTER = 'A'
JUDGE_CHARACTER = chr(1)
FILLED_CHARACTER = 'X'

BUCKET_CHARACTER_TO_BOX_CHARACTERS = {
    '}': [str(x) for x in range(1, 4)],
    ')': [str(x) for x in range(4, 7)],
    ']': [str(x) for x in range(7, 10)]
}

BUCKET_CHARACTER_TO_DISPLAYED_BOX_CHARACTER = {
    '}': '{',
    ')': '(',
    ']': '['
}

ALL_BUCKET_CHARACTERS = ['}', ')', ']']
ALL_CHARACTERS = [PLAYER_CHARACTER, JUDGE_CHARACTER, BACKGROUND_CHARACTER, WALL_CHARACTER, FILLED_CHARACTER] + ALL_BUCKET_CHARACTERS
ALL_BOX_CHARACTERS = []

for bucket_character in ALL_BUCKET_CHARACTERS:
    box_characters = BUCKET_CHARACTER_TO_BOX_CHARACTERS[bucket_character]
    ALL_BOX_CHARACTERS += box_characters
    ALL_CHARACTERS += box_characters

IMPASSABLE_CHARACTERS = list(set(ALL_CHARACTERS) - set([BACKGROUND_CHARACTER]))

# Displayed characters
DISPLAYED_CHARACTER_MAP = {}
ALL_DISPLAYED_CHARACTERS = [PLAYER_CHARACTER, BACKGROUND_CHARACTER, WALL_CHARACTER, FILLED_CHARACTER] + ALL_BUCKET_CHARACTERS
for bucket_character in ALL_BUCKET_CHARACTERS:
    box_characters = BUCKET_CHARACTER_TO_BOX_CHARACTERS[bucket_character]
    box_displayed_character = BUCKET_CHARACTER_TO_DISPLAYED_BOX_CHARACTER[bucket_character]
    ALL_DISPLAYED_CHARACTERS.append(box_displayed_character)

    for char in box_characters:
        DISPLAYED_CHARACTER_MAP[char] = box_displayed_character


# Knowledge graph
def make_warehouse_knowledge_graph():
    G = nx.DiGraph()

    # node features
    for i, char in enumerate(ALL_DISPLAYED_CHARACTERS):
        feature = np.zeros((len(ALL_DISPLAYED_CHARACTERS),), dtype=np.float32)
        feature[i] = 1
        G.add_node(ord(char), node_feature=feature)

    # relationships
    # We have three edge features: 'fills', 'pushes', and 'impassable'
    fills_feature = np.array([1, 0, 0], dtype=np.float32)
    pushes_feature = np.array([0, 1, 0], dtype=np.float32)
    impassable_feature = np.array([0, 0, 1], dtype=np.float32)

    # Fills and pushes
    for bucket_char, box_char in BUCKET_CHARACTER_TO_DISPLAYED_BOX_CHARACTER.items():
        G.add_edge(ord(box_char), ord(bucket_char), edge_feature=fills_feature)
        G.add_edge(ord(PLAYER_CHARACTER), ord(box_char), edge_feature=pushes_feature)

    # Impassable
    for bucket_char in ALL_BUCKET_CHARACTERS:
        G.add_edge(ord(PLAYER_CHARACTER), ord(bucket_char), edge_feature=impassable_feature)
    G.add_edge(ord(PLAYER_CHARACTER), ord(WALL_CHARACTER), edge_feature=impassable_feature)

    entities = [ord(char) for char in ALL_DISPLAYED_CHARACTERS]
    G = nx.relabel_nodes(G, {entity: i for i, entity in enumerate(entities)})

    return entities, G, len(ALL_DISPLAYED_CHARACTERS), 3


class PlayerSprite(MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(corner, position, character, set(IMPASSABLE_CHARACTERS) - set([character]))

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        if actions == 0:
            self._north(board, the_plot)
        elif actions == 1:
            self._south(board, the_plot)
        elif actions == 2:
            self._west(board, the_plot)
        elif actions == 3:
            self._east(board, the_plot)


class BoxSprite(MazeWalker):
    def __init__(self, corner, position, character):
        impassable = set(IMPASSABLE_CHARACTERS)
        impassable -= set([character])

        for bucket_character, box_characters in BUCKET_CHARACTER_TO_BOX_CHARACTERS.items():
            if character in box_characters:
                impassable -= set([bucket_character])
        super(BoxSprite, self).__init__(corner, position, character, impassable)

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        rows, cols = self.position
        if actions == 0:
            if layers['A'][rows + 1, cols]:
                self._north(board, the_plot)
        elif actions == 1:
            if layers['A'][rows - 1, cols]:
                self._south(board, the_plot)
        elif actions == 2:
            if layers['A'][rows, cols + 1]:
                self._west(board, the_plot)
        elif actions == 3:
            if layers['A'][rows, cols - 1]:
                self._east(board, the_plot)


class BucketDrape(things.Drape):
    def __init__(self, curtain, character):
        super(BucketDrape, self).__init__(curtain, character)
        self.last_buckets_filled = 0

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        box_characters = BUCKET_CHARACTER_TO_BOX_CHARACTERS[self.character]
        filled_mask = np.zeros_like(self.curtain)
        for char in box_characters:
            if char in all_things:
                filled_mask[all_things[char].position] = True
        filled_mask = np.logical_and(self.curtain, filled_mask)
        the_plot['filled_mask_{}'.format(self.character)] = filled_mask

        num_filled = np.sum(filled_mask)
        num_filled_change = num_filled - self.last_buckets_filled
        if num_filled_change != 0:
            the_plot.add_reward(3 * num_filled_change)
        if num_filled == self.curtain.sum():
            the_plot['all_filled_{}'.format(self.character)] = True
        else:
            the_plot['all_filled_{}'.format(self.character)] = False
        self.last_buckets_filled = num_filled


class FilledDrape(things.Drape):
    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        self.curtain.fill(False)
        for char in ALL_BUCKET_CHARACTERS:
            filled_mask = the_plot['filled_mask_{}'.format(char)]
            np.logical_or(self.curtain, filled_mask, out=self.curtain)


class JudgeDrape(things.Drape):
    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is not None:
            num_all_filled = 0
            for bucket_character in ALL_BUCKET_CHARACTERS:
                if the_plot['all_filled_{}'.format(bucket_character)]:
                    num_all_filled += 1
            if num_all_filled == len(ALL_BUCKET_CHARACTERS):
                the_plot.terminate_episode()
            else:
                the_plot.add_reward(-0.01)


def make_warehouse_env(artfile, encode_onehot=False):
    art, art_pycolab_characters, (height, width) = load_art(artfile)

    box_characters_in_game = set()
    for char in art_pycolab_characters:
        if char in ALL_BOX_CHARACTERS:
            box_characters_in_game.add(char)
    box_characters_in_game = list(box_characters_in_game)

    def make_pycolab_game():
        sprites = {}
        sprites[PLAYER_CHARACTER] = PlayerSprite
        for box_char in box_characters_in_game:
            sprites[box_char] = BoxSprite

        drapes = {}
        drapes[JUDGE_CHARACTER] = JudgeDrape
        drapes[FILLED_CHARACTER] = FilledDrape
        for char in ALL_BUCKET_CHARACTERS:
            drapes[char] = BucketDrape

        update_schedule = []
        update_schedule.append(box_characters_in_game)
        update_schedule.append(ALL_BUCKET_CHARACTERS)
        update_schedule.append([PLAYER_CHARACTER, FILLED_CHARACTER, JUDGE_CHARACTER])

        return ascii_art.ascii_art_to_game(
            art,
            what_lies_beneath=BACKGROUND_CHARACTER,
            sprites=sprites,
            drapes=drapes,
            update_schedule=update_schedule
        )

    env = PycolabMazeEnv(make_game_function=make_pycolab_game,
                         num_actions=4,
                         height=height, width=width,
                         character_map=DISPLAYED_CHARACTER_MAP)

    if encode_onehot:
        values = [ord(char) for char in ALL_DISPLAYED_CHARACTERS]
        env = OneHotEnv(env, values)
    return env


def make_sampled_warehouse_env(artfiles, encode_onehot=False):
    envs = [make_warehouse_env(artfile, encode_onehot=encode_onehot) for artfile in artfiles]
    env = SampleEnv(envs)
    return env
