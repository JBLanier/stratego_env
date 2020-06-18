import random

import h5py
import numpy as np

from stratego_env.game import stratego_procedural_env
from stratego_env.game.enums import GameVersions
from stratego_env.game.stratego_procedural_env import StrategoProceduralEnv
from stratego_env.game.stratego_procedural_impl import INT_DTYPE_NP
from stratego_env.game.stratego_procedural_impl import StateData


def _create_random_initial_piece_map(game_version_config: dict):
    correct_board_shape = (game_version_config['rows'], game_version_config['columns'])
    initial_piece_map = np.zeros(shape=correct_board_shape, dtype=INT_DTYPE_NP)

    valid_piece_locations = [(row, col)
                             for row in range(game_version_config['initial_state_usable_rows'])
                             for col in range(game_version_config['columns'])]

    random.shuffle(valid_piece_locations)

    idx_in_valid_piece_locations = 0
    for piece_type, amount_allowed in game_version_config['piece_amounts'].items():

        for _ in range(amount_allowed):
            initial_piece_map[valid_piece_locations[idx_in_valid_piece_locations]] = piece_type.value
            idx_in_valid_piece_locations += 1

    return initial_piece_map


def get_random_initial_state_fn(base_env: StrategoProceduralEnv, game_version_config: dict):
    def random_initial_state():

        correct_board_shape = (game_version_config['rows'], game_version_config['columns'])

        obstacle_map = np.zeros(shape=correct_board_shape, dtype=INT_DTYPE_NP)
        for obstacle_location in game_version_config['obstacle_locations']:
            obstacle_map[obstacle_location] = 1

        piece_maps = []

        for i in range(2):
            piece_maps.append(_create_random_initial_piece_map(game_version_config))

        return base_env.create_initial_state(
            obstacle_map=obstacle_map,
            player_1_initial_piece_map=piece_maps[0],
            player_2_initial_piece_map=piece_maps[1],
            max_turns=game_version_config['max_turns'])

    return random_initial_state


# -------------------------------------------------------------------------

# class SP(Enum):
#     """Stratego Pieces
#     SPY (1) through Marshal (10) are defined by their rank.
#     FLAG (11), BOMB (12), and UNKNOWN (13) are defined with special values.
#     """
#     NOPIECE = INT_DTYPE_NP(0)
#     SPY = INT_DTYPE_NP(1)
#     SCOUT = INT_DTYPE_NP(2)
#     MINER = INT_DTYPE_NP(3)
#     SERGEANT = INT_DTYPE_NP(4)
#     LIEUTENANT = INT_DTYPE_NP(5)
#     CAPTAIN = INT_DTYPE_NP(6)
#     MAJOR = INT_DTYPE_NP(7)
#     COLONEL = INT_DTYPE_NP(8)
#     GENERAL = INT_DTYPE_NP(9)
#     MARSHALL = INT_DTYPE_NP(10)
#     FLAG = INT_DTYPE_NP(11)
#     BOMB = INT_DTYPE_NP(12)
#     UNKNOWN = INT_DTYPE_NP(13)

# --- Counts from left to right up to down, even in in opponent ---

# ------------------------------------
#  Double check this again
# ------------------------------------

# Player 1. Left side. Bottom board.
# B: BOMB
# C: SPY
# D: SCOUT
# E: MINER
# F: SERGEANT
# G: LIEUTENANT
# H: CAPTAIN
# I: MAJOR
# J: COLONEL
# K: GENERAL
# L: MARSHALL
# M: FLAG

# Player 2. Right side. Top board.
# N: BOMB
# O: SPY
# P: SCOUT
# Q: MINER
# R: SERGEANT
# S: LIEUTENANT
# T: CAPTAIN
# U: MAJOR
# V: COLONEL
# W: GENERAL
# X: MARSHALL
# Y: FLAG

# A: NOPIECE

def create_random_board(game_version_config: dict):
    correct_board_shape = (game_version_config['rows'], game_version_config['columns'])
    initial_piece_map = np.zeros(shape=correct_board_shape, dtype=INT_DTYPE_NP)

    valid_piece_locations = [(row, col)
                             for row in range(game_version_config['initial_state_usable_rows'])
                             for col in range(game_version_config['columns'])]

    random.shuffle(valid_piece_locations)

    idx_in_valid_piece_locations = 0
    for piece_type, amount_allowed in game_version_config['piece_amounts'].items():

        for _ in range(amount_allowed):
            initial_piece_map[valid_piece_locations[idx_in_valid_piece_locations]] = piece_type.value
            idx_in_valid_piece_locations += 1

    return initial_piece_map


def create_board_from_string(game_version_config: dict):
    correct_board_shape = (game_version_config['rows'], game_version_config['columns'])
    initial_piece_map = np.zeros(shape=correct_board_shape, dtype=INT_DTYPE_NP)

    valid_piece_locations = [(row, col)
                             for row in range(game_version_config['initial_state_usable_rows'])
                             for col in range(game_version_config['columns'])]

    random.shuffle(valid_piece_locations)

    idx_in_valid_piece_locations = 0
    for piece_type, amount_allowed in game_version_config['piece_amounts'].items():

        for _ in range(amount_allowed):
            initial_piece_map[valid_piece_locations[idx_in_valid_piece_locations]] = piece_type.value
            idx_in_valid_piece_locations += 1

    return initial_piece_map


def convert_letter_to_num_left(letter):
    if letter == 'A':
        return 0
    if letter == 'B':
        return 12
    elif letter == 'C':
        return 1
    elif letter == 'D':
        return 2
    elif letter == 'E':
        return 3
    elif letter == 'F':
        return 4
    elif letter == 'G':
        return 5
    elif letter == 'H':
        return 6
    elif letter == "I":
        return 7
    elif letter == "J":
        return 8
    elif letter == "K":
        return 9
    elif letter == "L":
        return 10
    elif letter == "M":
        return 11


def convert_letter_to_num_right(letter):
    if letter == 'A':
        return 0
    if letter == 'N':
        return 12
    elif letter == 'O':
        return 1
    elif letter == 'P':
        return 2
    elif letter == 'Q':
        return 3
    elif letter == 'R':
        return 4
    elif letter == 'S':
        return 5
    elif letter == 'T':
        return 6
    elif letter == "U":
        return 7
    elif letter == "V":
        return 8
    elif letter == "W":
        return 9
    elif letter == "X":
        return 10
    elif letter == "Y":
        return 11


# returns a 4x10 2D array of the first four rows
# these are the actual positions of the player
# by is left and right I think they mean
# that left is player 1 on top
# and right is player 2 on bottom
def convert(game_line, is_left):
    row = 0
    arr = []
    i = 0
    temp_inner_arr = []
    for piece in game_line:
        if piece == "A":
            temp_inner_arr.append(0)
        else:
            # convert from letters (like in the data) to numbers
            if is_left:
                temp_inner_arr.append(convert_letter_to_num_left(piece))
            else:
                temp_inner_arr.append(convert_letter_to_num_right(piece))
        i += 1
        if i % 10 == 0:
            arr.append(temp_inner_arr)
            temp_inner_arr = []
    return arr


# Add parameters that take in string in format of "LDAAAAAAAAKCAAAAAAAABEAAAAAAAAMDAAAAAAAA"
# calls "convert" to convert with the string to convert it to the first four rows of the board
# adds 0 to the rest of the board
def create_initial_positions(player1_string, player2_string, game_version_config: dict):
    correct_board_shape = (game_version_config['rows'], game_version_config['columns'])
    initial_piece_map1 = np.zeros(shape=correct_board_shape, dtype=INT_DTYPE_NP)
    initial_piece_map2 = np.zeros(shape=correct_board_shape, dtype=INT_DTYPE_NP)
    positions = np.asarray([initial_piece_map1, initial_piece_map2])

    first = convert(player1_string, True)
    # In my data I converted the player2 strings to be the same format as the player 1 strings
    second = convert(player2_string, True)  # False)
    firstFours = [first, second]

    for player in range(2):
        for i in range(10):
            for j in range(10):
                if i < 4:
                    positions[player][i][j] = firstFours[player][i][j]

    # for some reason, it is flipped 180 degrees over y axis so we need to invert
    for i in range(10):
        j = 0
        k = 9
        while j < k:
            positions[0][i][j], positions[0][i][k] = positions[0][i][k], positions[0][i][j]
            j += 1
            k -= 1

    # rotates 180 over x axis for player 2
    for i in range(2):
        for k in range(10):
            positions[i][0][k], positions[i][3][k] = positions[i][3][k], positions[i][0][k]
            positions[i][1][k], positions[i][2][k] = positions[i][2][k], positions[i][1][k]

    positions = positions[:, :, ::-1]

    return positions


def create_game_from_data(player1_string, player2_string, game_version_config, procedural_env=None):
    correct_board_shape = (game_version_config['rows'], game_version_config['columns'])

    if procedural_env is None:
        procedural_env = stratego_procedural_env.StrategoProceduralEnv(*correct_board_shape)

    obstacle_map = np.zeros(shape=correct_board_shape, dtype=INT_DTYPE_NP)
    for obstacle_location in game_version_config['obstacle_locations']:
        obstacle_map[obstacle_location] = 1.0

    piece_maps = create_initial_positions(player1_string, player2_string, game_version_config)

    max_turns = game_version_config['max_turns']

    game = procedural_env.create_initial_state(
        obstacle_map=np.array(obstacle_map),
        player_1_initial_piece_map=np.array(piece_maps[0]),
        player_2_initial_piece_map=np.array(piece_maps[1]),
        max_turns=int(max_turns))

    return game


def get_random_human_init_fn(game_version, game_version_config, procedural_env):
    if game_version in [GameVersions.STANDARD, GameVersions.SHORT_STANDARD, GameVersions.MEDIUM_STANDARD]:
        from stratego_env.game.inits.standard_human_inits import STANDARD_INITS as HUMAN_INITS
    elif game_version in [GameVersions.BARRAGE, GameVersions.SHORT_BARRAGE]:
        from stratego_env.game.inits.barrage_human_inits import BARRAGE_INITS as HUMAN_INITS
    else:
        raise ValueError("Human inits not supported with {} game version".format(game_version))

    def random_human_init():
        player_1_string = np.random.choice(HUMAN_INITS)
        player_2_string = np.random.choice(HUMAN_INITS)

        return create_game_from_data(player1_string=player_1_string, player2_string=player_2_string,
                                     game_version_config=game_version_config, procedural_env=procedural_env)

    return random_human_init


def load_h5(fname, debug=0, dtype=None, key=None, raise_on_failure=True, num_samples_to_load=None, read_offset=0):
    """load h5 file
        if <key> is str: will load h5file[key]
        if <key> is list: will load h5file[key[0]][key[1]][key[2]]....
    """
    hfile = h5py.File(fname, 'r')
    if debug:
        print(fname)
        print(hfile.keys())
    if key is None:
        try:
            key = list(hfile.keys())[0]
            xx = hfile[key]
        except:
            print("\nload_h5()::ERROR: File is not h5 / is empty <<", fname, ">>\n")
            if raise_on_failure:
                assert 0, "Cannot read file (not h5/empty)"
            return None
    elif isinstance(key, str):
        xx = hfile[key]
    elif isinstance(key, list):
        xx = hfile
        for k in key:
            xx = xx[k]
    else:
        raise ValueError()
    if isinstance(xx, h5py.Group):
        xx = dict(xx)
        xx = xx[list(xx.keys())[0] if key is None else key]
    if debug:
        print("type is:", xx.dtype, "shape is:", xx.shape)

    if read_offset == -1:
        # random offset
        read_offset = np.random.randint(low=0, high=len(xx))

    if num_samples_to_load is None:
        if read_offset == 0:
            dat = np.asarray(xx, dtype=dtype if dtype is not None else xx.dtype)
        else:
            dat = np.asarray(xx[read_offset:], dtype=dtype if dtype is not None else xx.dtype)
    else:
        dat = np.asarray(xx[read_offset:read_offset + num_samples_to_load],
                         dtype=dtype if dtype is not None else xx.dtype)
    if debug:
        print(np.shape(dat))
        print(dat.dtype)
        print()
    hfile.close()
    return dat, read_offset


def get_random_curriculum_init_fn(inits_path, max_turns):
    def random_human_init():
        state, offset = load_h5(fname=inits_path, key='state', num_samples_to_load=1, read_offset=-1)
        winner, w_offset = np.squeeze(
            load_h5(fname=inits_path, key='winner', num_samples_to_load=1, read_offset=offset))
        assert offset == w_offset
        state = np.squeeze(state)
        winner = int(np.squeeze(winner))
        state[StateData.TURN_COUNT.value] = 0.0
        state[StateData.MAX_TURNS.value] = max_turns

        return state, winner

    return random_human_init
