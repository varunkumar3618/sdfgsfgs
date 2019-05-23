import numpy as np
import networkx as nx
from pycolab import ascii_art, things
from pycolab.prefab_parts.sprites import MazeWalker

from graphrl.environments.pycolab_wrappers import load_art, PycolabMazeEnv, OneHotEnv

# Characters in the pycolab game

BACKGROUND_CHARACTER = ' '

WALL_CHARACTER = '+'

PLAYER_CHARACTER = 'A'
JUDGE_CHARACTER = chr(1)
FILLED_CHARACTER = 'X'


BUCKET_CHARACTERS = [chr(x) for x in range(ord('B'), ord('W') + 1)]
BOX_CHARACTERS = [chr(x) for x in range(ord('b'), ord('w') + 1)] + [str(x) for x in range(10)]


ALL_CHARACTERS = [BACKGROUND_CHARACTER, WALL_CHARACTER, PLAYER_CHARACTER, JUDGE_CHARACTER, FILLED_CHARACTER]
ALL_CHARACTERS = ALL_CHARACTERS + BUCKET_CHARACTERS
ALL_CHARACTERS = ALL_CHARACTERS + BOX_CHARACTERS


def maybe_initialize(the_plot):
    if 'filled_mask' not in the_plot:
        the_plot['filled_mask'] = {}
    if 'all_filled' not in the_plot:
        the_plot['all_filled'] = {}


class PlayerSprite(MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(corner, position, character, set(ALL_CHARACTERS) - set([character, BACKGROUND_CHARACTER]))

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        maybe_initialize(the_plot)

        if actions == 0:
            self._north(board, the_plot)
        elif actions == 1:
            self._south(board, the_plot)
        elif actions == 2:
            self._west(board, the_plot)
        elif actions == 3:
            self._east(board, the_plot)


class BoxSprite(MazeWalker):
    def __init__(self, corner, position, character, bucket_to_boxes):
        impassable = set(ALL_CHARACTERS) - set([character, BACKGROUND_CHARACTER])

        for bucket, boxes in bucket_to_boxes.items():
            if character in boxes and bucket in impassable:
                impassable.remove(bucket)

        super(BoxSprite, self).__init__(corner, position, character, impassable)

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        maybe_initialize(the_plot)

        rows, cols = self.position
        if actions == 0:
            if layers[PLAYER_CHARACTER][rows + 1, cols]:
                self._north(board, the_plot)
        elif actions == 1:
            if layers[PLAYER_CHARACTER][rows - 1, cols]:
                self._south(board, the_plot)
        elif actions == 2:
            if layers[PLAYER_CHARACTER][rows, cols + 1]:
                self._west(board, the_plot)
        elif actions == 3:
            if layers[PLAYER_CHARACTER][rows, cols - 1]:
                self._east(board, the_plot)


class BucketDrape(things.Drape):
    def __init__(self, curtain, character, bucket_to_boxes):
        super(BucketDrape, self).__init__(curtain, character)
        self.last_buckets_filled = 0
        self.bucket_to_boxes = bucket_to_boxes

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        maybe_initialize(the_plot)

        filled_mask = np.zeros_like(self.curtain)
        for box_char in self.bucket_to_boxes[self.character]:
            if box_char in all_things:
                filled_mask[all_things[box_char].position] = True
        filled_mask = np.logical_and(self.curtain, filled_mask)
        the_plot['filled_mask'][self.character] = filled_mask

        num_filled = np.sum(filled_mask)
        num_filled_change = num_filled - self.last_buckets_filled
        if num_filled_change != 0:
            the_plot.add_reward(3 * num_filled_change)
        if num_filled == self.curtain.sum():
            the_plot['all_filled'][self.character] = True
        else:
            the_plot['all_filled'][self.character] = False
        self.last_buckets_filled = num_filled


class FilledDrape(things.Drape):
    def __init__(self, curtain, character, buckets):
        super(FilledDrape, self).__init__(curtain, character)
        self.buckets = buckets

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        maybe_initialize(the_plot)

        self.curtain.fill(False)
        for char in self.buckets:
            if char in all_things:
                filled_mask = the_plot['filled_mask'][char]
                np.logical_or(self.curtain, filled_mask, out=self.curtain)


class JudgeDrape(things.Drape):
    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is not None:
            maybe_initialize(the_plot)
            should_continue = False
            for bucket_all_filled in the_plot['all_filled'].values():
                if not bucket_all_filled:
                    should_continue = True
                    break
            if should_continue:
                the_plot.add_reward(-0.01)
            else:
                the_plot.terminate_episode()


def make_warehouse_env(artfile, boxes, buckets, bucket_to_boxes, character_map={}, encode_onehot=False):
    art, art_characters, (height, width) = load_art(artfile)

    def make_pycolab_game():
        sprites = {}
        sprites[PLAYER_CHARACTER] = PlayerSprite

        game_boxes = [char for char in BOX_CHARACTERS if char in art_characters]
        game_buckets = [char for char in BUCKET_CHARACTERS if char in art_characters]

        for char in game_boxes:
            sprites[char] = ascii_art.Partial(BoxSprite, bucket_to_boxes=bucket_to_boxes)

        drapes = {}
        drapes[JUDGE_CHARACTER] = JudgeDrape
        drapes[FILLED_CHARACTER] = ascii_art.Partial(FilledDrape, buckets=list(bucket_to_boxes.keys()))
        for char in game_buckets:
            drapes[char] = ascii_art.Partial(BucketDrape, bucket_to_boxes=bucket_to_boxes)

        update_schedule = []
        update_schedule.append(game_boxes)
        update_schedule.append(game_buckets)
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
                         character_map=character_map)

    entities = make_warehouse_knowledge_graph(boxes, buckets, bucket_to_boxes, character_map=character_map)[0]
    if encode_onehot:
        env = OneHotEnv(env, entities)

    return env


def make_warehouse_knowledge_graph(boxes, buckets, bucket_to_boxes, character_map={}, no_edges=False, fully_connected=False, fully_connected_distinct=False, use_negative_fills=False, use_background_entity=False, same_edge_feats=False, use_agent_filled=False):
    '''
    boxes, buckets, and bucket_to_boxes are specified using the original characters, i.e. before character_map is applied
    '''
    G = nx.DiGraph()

    world_raw_characters = [BACKGROUND_CHARACTER, WALL_CHARACTER, PLAYER_CHARACTER, FILLED_CHARACTER]
    world_raw_characters = world_raw_characters + boxes + buckets
    if use_background_entity:
        world_raw_characters = world_raw_characters + [BACKGROUND_CHARACTER]

    world_displayed_characters = [character_map.get(char, char) for char in world_raw_characters]
    world_displayed_characters = list(set(world_displayed_characters))
    world_displayed_characters = list(sorted(world_displayed_characters))

    for i, displayed_char in enumerate(world_displayed_characters):
        feature = np.zeros((len(world_displayed_characters),), dtype=np.float32)
        feature[i] = 1
        G.add_node(displayed_char, node_feature=feature)

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
                    feature = np.zeros((num_edge_features,), dtype=np.float32)
                    idx = i * (len(G.nodes) - 1) + j
                    feature[idx] = 1
                    G.add_edge(n1, n2, edge_feature=feature)
    else:
        feature_types = {
            'fills': 0,
            'pushes': 1,
            'impassable': 2
        }
        num_edge_features = 3
        if use_negative_fills:
            feature_types['no_fills'] = num_edge_features
            num_edge_features += 1
        if use_agent_filled:
            feature_types['agent_filled'] = num_edge_features
            num_edge_features += 1

        features = {}
        for i, (k, v) in enumerate(feature_types.items()):
            feature = np.zeros((num_edge_features,), dtype=np.float32)
            if same_edge_feats:
                v = 0
            feature[v] = 1
            features[k] = feature

        displayed_bucket_to_boxes = {}
        for bucket_raw_char in bucket_to_boxes.keys():
            bucket_displayed_char = character_map.get(bucket_raw_char, bucket_raw_char)

            if bucket_displayed_char not in displayed_bucket_to_boxes:
                displayed_bucket_to_boxes[bucket_displayed_char] = set()

            for box_raw_char in bucket_to_boxes[bucket_raw_char]:
                box_displayed_char = character_map.get(box_raw_char, box_raw_char)
                displayed_bucket_to_boxes[bucket_displayed_char].add(box_displayed_char)

        displayed_bucket_to_boxes = {key: list(val) for key, val in displayed_bucket_to_boxes.items()}

        # Fills
        for bucket_displayed_char in displayed_bucket_to_boxes:
            for box_displayed_char in displayed_bucket_to_boxes[bucket_displayed_char]:
                G.add_edge(box_displayed_char, bucket_displayed_char, edge_feature=features['fills'])

        # Agent filled
        if use_agent_filled:
            G.add_edge(character_map.get(PLAYER_CHARACTER, PLAYER_CHARACTER), character_map.get(FILLED_CHARACTER, FILLED_CHARACTER), edge_feature=features['agent_filled'])

        # Doesn't fill
        displayed_boxes = list(set([character_map.get(char, char) for char in boxes]))
        if use_negative_fills:
            for bucket_displayed_char in displayed_bucket_to_boxes:
                for box_displayed_char in displayed_boxes:
                    if box_displayed_char not in displayed_bucket_to_boxes[bucket_displayed_char]:
                        G.add_edge(box_displayed_char, bucket_displayed_char, edge_feature=features['no_fills'])

        # Pushes
        for box_displayed_char in displayed_boxes:
            G.add_edge(character_map.get(PLAYER_CHARACTER, PLAYER_CHARACTER), box_displayed_char, edge_feature=features['pushes'])

        # Impassable
        displayed_buckets = list(set([character_map.get(char, char) for char in buckets]))
        for bucket_displayed_char in displayed_buckets:
            G.add_edge(PLAYER_CHARACTER, bucket_displayed_char, edge_feature=features['impassable'])
        G.add_edge(character_map.get(PLAYER_CHARACTER, PLAYER_CHARACTER), character_map.get(WALL_CHARACTER, WALL_CHARACTER), edge_feature=features['impassable'])

    G = nx.relabel_nodes(G, {entity: ord(entity) for entity in G.nodes})
    return [ord(char) for char in world_displayed_characters], G, len(world_displayed_characters), num_edge_features
