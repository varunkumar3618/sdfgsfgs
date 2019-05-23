from pycolab import ascii_art, things
from pycolab.prefab_parts.sprites import MazeWalker

from graphrl.environments.pycolab_wrappers import load_art, OneHotEnv, PycolabMazeEnv, OneHotWithEqualFeatsEnv


class PlayerSprite(MazeWalker):
    def __init__(self, corner, position, character):
        super(PlayerSprite, self).__init__(corner, position, character, '+')

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        the_plot['something_happened'] = False
        if 'coins_over' not in the_plot:
            the_plot['coins_over'] = set()
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


class CoinsDrape(things.Drape):
    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        player_pos = all_things['A'].position
        if self.curtain[player_pos]:
            the_plot['something_happened'] = True
            the_plot.add_reward(3)
            self.curtain[player_pos] = False
        if self.curtain.sum() == 0:
            the_plot['coins_over'].add(self.character)


class PoisonDrape(things.Drape):
    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if actions is None:
            return
        player_pos = all_things['A'].position
        if self.curtain[player_pos]:
            the_plot['something_happened'] = True
            the_plot.add_reward(-3)
            the_plot.terminate_episode()


class JudgeDrape(things.Drape):
    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        if not the_plot['something_happened']:
            the_plot.add_reward(-0.01)
        if len(the_plot['coins_over']) == 4:
            the_plot.terminate_episode()


def make_coins_maze_v2_env(artfile, encode_onehot=False, add_equal_feats=False):
    art, characters, (height, width) = load_art(artfile)

    def make_pycolab_game():
        return ascii_art.ascii_art_to_game(
            art,
            what_lies_beneath=' ',
            sprites={
                'A': PlayerSprite,
            },
            drapes={
                '1': CoinsDrape,
                '2': CoinsDrape,
                '3': CoinsDrape,
                '4': CoinsDrape,
                'E': PoisonDrape,
                'X': JudgeDrape
            },
            update_schedule=[['A'], ['1', '2', '3', '4', 'E'], ['X']]
        )

    env = PycolabMazeEnv(make_game_function=make_pycolab_game, num_actions=4, height=height, width=width)

    if encode_onehot:
        values = [ord(char) for char in characters]
        if add_equal_feats:
            equal_values = [[ord(char) for char in ['1', '2', '3', '4']]]
            env = OneHotWithEqualFeatsEnv(env, values, equal_values)
        else:
            env = OneHotEnv(env, values)
    return env
