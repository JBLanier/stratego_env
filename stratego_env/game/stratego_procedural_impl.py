from enum import Enum
from typing import Tuple

import numpy as np
from numba import int64, jit, types, boolean, float32

"""
All of these functions are also available as methods in StrategoProceduralEnv.
The functions defined here are not intended to be called directly. 
They are intended to be used through StrategoProceduralEnv.

This logic is JIT compiled with Numba. You will encounter a hang the first time you run this code while it compiles.
Afterward, it will be cached and you won't have to wait for it to compile.
"""

"""Internal State Breakdown:
shape = (layers, rows, columns)

state layers are defined by name in the StateLayers enum.

layers:
0: player 1 ground truth pieces (contents are SP enum types and CAN'T be UNKNOWN)
1: player 2 ground truth pieces (contents are SP enum types and CANT'T be UNKNOWN)
2: obstacle map (contents are 1 for 'obstacle at this space', 0 otherwise)

3: player 1 pieces from player 2's partially observable perspective (contents are SP enum types and CAN be UNKNOWN)
4: player 2 pieces from player 1's partially observable perspective (contents are SP enum types and CAN be UNKNOWN)

5: scalar data...
    state[3,0,0]: turn count (initial turn when no player has moved is 0, increments by 1 with each player action)
    state[3,0,1]: whether game is over (0 for False, 1 for True)
    state[3,0,2]: winner of the game if the game is over (0 for tie, 1 for player 1 won, -1 for player 2 won) (meaningless if game isn't over)
    state[3,1,0]: max turn count (after this many turns have happened, game is over and tied. We may or may not consider this kind of ending as invalid)
    state[3,1,1]: whether the game ending can be considered invalid (0 for False, 1 for True) (meaningless if game isn't over)

6: player 1 recent moves map for enforcing 2-squares rule against oscillating pieces back and forth (contents are RecentMoves enum types)
    This layer (and thus move tracking) is wiped to zeros anytime player 1 makes an attack.
    (0 for no move made here recently)
    (1 for the most recent piece to move came from here)
    (-1 for the most recent piece to move is now here)
    (-2 for the most recent piece to move is now here, and this is also the spot that it was in 2 turns ago.
       This piece cannot go to a spot marked as 1 and then return here.
    (-3 for the most recent piece to move is now here, and this is also the spot that it was in 2 turns ago.
       This piece cannot double back again (The piece here can't go to a spot marked with 1).

7: player 2 recent moves map (same rules apply)

8-19: player 1 captured pieces maps. Each piece is its own layer. (Layers are identified by _get_player_captured_piece_layer)
    These pieces were owned by player 1 but captured (and type revealed) by player 2.
    (0 for no piece of this type was ever captured at this location)
    (1 for 1 piece of the this type was captured at this location)
    (2 for 2 pieces of this type were captured at this location)
    etc.

20-31: player 2 captured pieces maps (same rules apply) (Layers are identified by _get_player_captured_piece_layer)

32: player 1 still pieces (from player 2's perspective), 1 for piece here that has never moved, 0 otherwise
33: player 2 still pieces (from player 1's perspective), 1 for piece here that has never moved, 0 otherwise

"""

INT_DTYPE_NP = np.int64
INT_DTYPE_JIT = int64

FLOAT_DTYPE_NP = np.float32
FLOAT_DTYPE_JIT = float32


class StateLayers(Enum):
    # Fully Observable Piece Locations
    PLAYER_1_PIECES = INT_DTYPE_NP(0)
    PLAYER_2_PIECES = INT_DTYPE_NP(1)

    OBSTACLES = INT_DTYPE_NP(2)

    # PO = Partially Observable
    PLAYER_1_PO_PIECES = INT_DTYPE_NP(3)
    PLAYER_2_PO_PIECES = INT_DTYPE_NP(4)

    # Scalar Data
    DATA = INT_DTYPE_NP(5)

    # 2-Squares rule tracking
    PLAYER_1_RECENT_MOVES = INT_DTYPE_NP(6)
    PLAYER_2_RECENT_MOVES = INT_DTYPE_NP(7)

    # Captured piece counts at each location (1 channel per piece type per player)
    # See _get_player_captured_piece_layer to get the correct layer for a piece type and player
    PLAYER_1_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(8)
    PLAYER_1_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(20)
    PLAYER_2_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(20)
    PLAYER_2_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(32)

    # Pieces for each player that have never moved
    PLAYER_1_STILL_PIECES = INT_DTYPE_NP(32)
    PLAYER_2_STILL_PIECES = INT_DTYPE_NP(33)


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_player_1_captured_piece_layer(piece_type):
    return INT_DTYPE_NP(7 + piece_type)


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_player_2_captured_piece_layer(piece_type):
    return INT_DTYPE_NP(19 + piece_type)


NUM_STATE_LAYERS = 34


class RecentMoves(Enum):
    # For tracking the 2-squares rule
    # (0 for no move made here recently)
    # (1 for the most recent piece to move came from here)
    # (-1 for the most recent piece to move is now here)
    # (-2 for the most recent piece to move is now here, and this is also the spot that it was in 2 turns ago.
    #   This piece cannot go to a spot marked as 1 and then return here.
    # (-3 for the most recent piece to move is now here, and this is also the spot that it was in 2 turns ago.
    #   This piece cannot double back again (The piece here can't go to a spot marked with 1).
    NODATA = INT_DTYPE_NP(0)
    JUST_CAME_FROM = INT_DTYPE_NP(1)
    JUST_ARRIVED = INT_DTYPE_NP(-1)
    JUST_ARRIVED_AND_NEXT_DOUBLE_BACK_IS_ILLEGAL = INT_DTYPE_NP(-2)
    JUST_ARRIVED_AND_CANT_DOUBLE_BACK = INT_DTYPE_NP(-3)


class StillPieces(Enum):
    # For still pieces state layers
    # 1 for piece has never moved
    # 0 for piece has moved before
    COULD_BE_FLAG = INT_DTYPE_NP(1)
    CANT_BE_FLAG = INT_DTYPE_NP(0)


class StateData(Enum):
    # specific slices for the state scalar data layer
    TURN_COUNT = np.s_[StateLayers.DATA.value, 0, 0]
    GAME_OVER = np.s_[StateLayers.DATA.value, 0, 1]
    WINNER = np.s_[StateLayers.DATA.value, 0, 2]
    MAX_TURNS = np.s_[StateLayers.DATA.value, 1, 0]
    ENDING_INVALID = np.s_[StateLayers.DATA.value, 1, 1]


class SP(Enum):
    """Stratego Pieces
    SPY (1) through Marshal (10) are defined by their rank.
    FLAG (11), BOMB (12), and UNKNOWN (13) are defined with special values.
    """
    NOPIECE = INT_DTYPE_NP(0)
    SPY = INT_DTYPE_NP(1)
    SCOUT = INT_DTYPE_NP(2)
    MINER = INT_DTYPE_NP(3)
    SERGEANT = INT_DTYPE_NP(4)
    LIEUTENANT = INT_DTYPE_NP(5)
    CAPTAIN = INT_DTYPE_NP(6)
    MAJOR = INT_DTYPE_NP(7)
    COLONEL = INT_DTYPE_NP(8)
    GENERAL = INT_DTYPE_NP(9)
    MARSHALL = INT_DTYPE_NP(10)
    FLAG = INT_DTYPE_NP(11)
    BOMB = INT_DTYPE_NP(12)
    UNKNOWN = INT_DTYPE_NP(13)


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_max_possible_actions_per_start_position(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP):
    return rows + columns  # scout can move anywhere horizontally or vertically,
    # in reality, 2 spots here will never be reached, but it's included for simplicity


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _player_index(player: INT_DTYPE_NP):
    # player 1 returns 0
    # player -1 returns 1
    return (player - 1) // -2


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _player_po_index(player: INT_DTYPE_NP):
    if player == 1:
        return StateLayers.PLAYER_1_PO_PIECES.value
    else:
        return StateLayers.PLAYER_2_PO_PIECES.value


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _player_moves_index(player: INT_DTYPE_NP):
    if player == 1:
        return StateLayers.PLAYER_1_RECENT_MOVES.value
    else:
        return StateLayers.PLAYER_2_RECENT_MOVES.value


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _player_still_pieces_index(player: INT_DTYPE_NP):
    if player == 1:
        return StateLayers.PLAYER_1_STILL_PIECES.value
    else:
        return StateLayers.PLAYER_2_STILL_PIECES.value


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_player_captured_piece_layer(player: INT_DTYPE_NP, piece_type: INT_DTYPE_NP):
    if player == 1:
        return _get_player_1_captured_piece_layer(piece_type)
    else:
        return _get_player_2_captured_piece_layer(piece_type)


@jit(INT_DTYPE_JIT[:, :, :](INT_DTYPE_JIT[:, :], INT_DTYPE_JIT[:, :], INT_DTYPE_JIT[:, :], INT_DTYPE_JIT),
     nopython=True, fastmath=True, cache=True)
def _create_initial_state(obstacle_map: np.ndarray,
                          player_1_initial_piece_map: np.ndarray,
                          player_2_initial_piece_map: np.ndarray,
                          max_turns: INT_DTYPE_NP):
    rows, columns = obstacle_map.shape

    state = np.zeros(shape=(NUM_STATE_LAYERS, rows, columns), dtype=INT_DTYPE_NP)
    state[StateLayers.PLAYER_1_PIECES.value] = player_1_initial_piece_map.copy()
    state[StateLayers.PLAYER_2_PIECES.value] = player_2_initial_piece_map[::-1, ::-1].copy()

    # all piece locations are marked as UNKNOWN
    player_1_po_pieces = player_1_initial_piece_map.copy()
    player_1_po_pieces = np.where(player_1_po_pieces != SP.NOPIECE.value, SP.UNKNOWN.value, player_1_po_pieces)
    state[StateLayers.PLAYER_1_PO_PIECES.value] = player_1_po_pieces

    # all piece locations are marked as UNKNOWN
    player_2_po_pieces = player_2_initial_piece_map[::-1, ::-1].copy()
    player_2_po_pieces = np.where(player_2_po_pieces != SP.NOPIECE.value, SP.UNKNOWN.value, player_2_po_pieces)
    state[StateLayers.PLAYER_2_PO_PIECES.value] = player_2_po_pieces

    # are pieces all initially marked as 'never moved'
    player_1_still_pieces = player_1_initial_piece_map.copy()
    player_1_still_pieces = np.where(player_1_still_pieces != SP.NOPIECE.value, StillPieces.COULD_BE_FLAG.value,
                                     player_1_still_pieces)
    state[StateLayers.PLAYER_1_STILL_PIECES.value] = player_1_still_pieces

    # are pieces all initially marked as 'never moved'
    player_2_still_pieces = player_2_initial_piece_map[::-1, ::-1].copy()
    player_2_still_pieces = np.where(player_2_still_pieces != SP.NOPIECE.value, StillPieces.COULD_BE_FLAG.value,
                                     player_2_still_pieces)
    state[StateLayers.PLAYER_2_STILL_PIECES.value] = player_2_still_pieces

    state[StateLayers.OBSTACLES.value] = obstacle_map.copy()

    state[StateData.MAX_TURNS.value] = max_turns

    return state


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_action_size(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP):
    return (rows * columns * _get_max_possible_actions_per_start_position(rows, columns)) + 1


@jit(types.UniTuple(INT_DTYPE_JIT, 3)(INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_spatial_action_size(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP):
    return (rows, columns, (rows - 1) * 2 + (columns - 1) * 2 + 1)


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT,
                   INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_action_1d_index_from_positions(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP, start_r: INT_DTYPE_NP,
                                        start_c: INT_DTYPE_NP,
                                        end_r: INT_DTYPE_NP, end_c: INT_DTYPE_NP,
                                        max_possible_actions_per_start_position: INT_DTYPE_NP):
    if end_r != start_r:
        # row changed by action
        end_position_offset = end_r
    else:
        # column changed by action
        end_position_offset = rows + end_c

    index = (((start_r * columns) + start_c) * max_possible_actions_per_start_position) + end_position_offset

    return index


@jit(types.UniTuple(INT_DTYPE_JIT, 3)(INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT,
                                      INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_action_spatial_index_from_positions(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP, start_r: INT_DTYPE_NP,
                                             start_c: INT_DTYPE_NP,
                                             end_r: INT_DTYPE_NP, end_c: INT_DTYPE_NP):
    col_dist = end_c - start_c
    row_dist = end_r - start_r

    if not (col_dist == 0 or row_dist == 0):
        print("_get_action_spatial_index_from_positions: diagonal move encountered")
        assert False

    max_row_move_distance_on_board = rows - 1
    max_col_move_distance_on_board = columns - 1

    # M row x column channels where M is the sum of all possible moves distances in each of the 4 directions.

    if row_dist > 0:
        channel_offset = 0
    elif row_dist < 0:
        channel_offset = max_row_move_distance_on_board
    elif col_dist > 0:
        channel_offset = 2 * max_row_move_distance_on_board
    elif col_dist < 0:
        channel_offset = 2 * max_row_move_distance_on_board + max_col_move_distance_on_board
    else:
        raise ValueError("move start position and end position are the same")

    # either row_dist of col_dist should be zero
    channel = channel_offset + abs(row_dist + col_dist) - 1

    return start_r, start_c, channel


@jit(types.UniTuple(INT_DTYPE_JIT, 4)(INT_DTYPE_JIT, INT_DTYPE_JIT, types.UniTuple(INT_DTYPE_JIT, 3)), nopython=True,
     fastmath=True, cache=True)
def _get_action_positions_from_spatial_index(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP, spatial_index):
    start_r, start_c, channel = spatial_index

    max_row_move_distance_on_board = rows - 1
    max_col_move_distance_on_board = columns - 1

    if channel < max_row_move_distance_on_board:
        end_r = start_r + (channel + 1)
        end_c = start_c
    elif channel < 2 * max_row_move_distance_on_board:
        end_r = start_r - ((channel - max_row_move_distance_on_board) + 1)
        end_c = start_c
    elif channel < 2 * max_row_move_distance_on_board + max_col_move_distance_on_board:
        end_r = start_r
        end_c = start_c + ((channel - 2 * max_row_move_distance_on_board) + 1)
    else:
        end_r = start_r
        end_c = start_c - ((channel - (2 * max_row_move_distance_on_board + max_col_move_distance_on_board)) + 1)

    return start_r, start_c, end_r, end_c


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT, INT_DTYPE_JIT, types.UniTuple(INT_DTYPE_JIT, 3), INT_DTYPE_JIT), nopython=True,
     fastmath=True, cache=True)
def _get_action_1d_index_from_spatial_index(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP,
                                            spatial_index: Tuple[INT_DTYPE_NP],
                                            max_possible_actions_per_start_position: INT_DTYPE_NP):
    start_r, start_c, end_r, end_c = _get_action_positions_from_spatial_index(rows=rows, columns=columns,
                                                                              spatial_index=spatial_index, )
    return _get_action_1d_index_from_positions(
        rows=rows, columns=columns, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c,
        max_possible_actions_per_start_position=max_possible_actions_per_start_position)


@jit(types.UniTuple(INT_DTYPE_JIT, 4)(INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT),
     nopython=True, fastmath=True, cache=True)
def _get_action_positions_from_1d_index(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP, action_index: INT_DTYPE_NP,
                                        action_size: INT_DTYPE_NP,
                                        max_possible_actions_per_start_position: INT_DTYPE_NP):
    if action_index == action_size - 1:
        print("-------------------------------------------")
        print("action index:")
        print(action_index)
        print("action size:")
        print(action_size)
        print("max_possible_actions_per_start_position:")
        print(max_possible_actions_per_start_position)
        print("rows:")
        print(rows)
        print("columns:")
        print(columns)
        raise ValueError("Action is a no-op so it doesn't translate to an actual action")

    start_r = (action_index // max_possible_actions_per_start_position) // columns
    start_c = (action_index // max_possible_actions_per_start_position) % columns

    end_position_offset = action_index % max_possible_actions_per_start_position

    if end_position_offset >= rows:
        # column changed by action
        end_c = end_position_offset - rows
        end_r = start_r
    else:
        # row changed by action
        end_r = end_position_offset
        end_c = start_c

    return start_r, start_c, end_r, end_c


@jit(types.UniTuple(INT_DTYPE_JIT, 3)(INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT),
     nopython=True, fastmath=True, cache=True)
def _get_action_spatial_index_from_1d_index(rows: INT_DTYPE_NP, columns: INT_DTYPE_NP, action_index: INT_DTYPE_NP,
                                            action_size: INT_DTYPE_NP,
                                            max_possible_actions_per_start_position: INT_DTYPE_NP):
    start_r, start_c, end_r, end_c = _get_action_positions_from_1d_index(
        rows=rows, columns=columns, action_index=action_index, action_size=action_size,
        max_possible_actions_per_start_position=max_possible_actions_per_start_position)

    return _get_action_spatial_index_from_positions(
        rows=rows, columns=columns, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c)


@jit(INT_DTYPE_JIT[:, :, :](INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_valid_moves_as_spatial_mask(state: np.ndarray, player: INT_DTYPE_NP):
    layers, rows, columns = state.shape

    spatial_representation_channels = (rows - 1) * 2 + (
            columns - 1) * 2 + 1  # + 1 for noop action at [0, 0, extra_noop_channel_at_end]

    valid_moves_mask = np.zeros(shape=(rows, columns, spatial_representation_channels), dtype=INT_DTYPE_NP)
    owned_pieces = state[_player_index(player=player)]
    enemy_pieces = state[_player_index(player=-player)]
    obstacles = state[StateLayers.OBSTACLES.value]
    recent_moves = state[_player_moves_index(player=player)]

    no_moves_available = True

    if not state[StateData.GAME_OVER.value]:

        for start_r in range(rows):
            for start_c in range(columns):
                piece_type = owned_pieces[start_r, start_c]

                if piece_type == 0 or piece_type == SP.FLAG.value or piece_type == SP.BOMB.value:
                    continue
                elif piece_type == SP.SCOUT.value:
                    # scouts and move any amount in each of the 4 directions until they hit something
                    # move out in each direction until we hit the edge or a piece/obstacle and incrementally add valid moves

                    # First handle vertical movement cases
                    row_directions = (INT_DTYPE_NP(1), INT_DTYPE_NP(-1))
                    for r_dir in row_directions:
                        r_offset = INT_DTYPE_NP(0)
                        end_c = start_c
                        while True:
                            r_offset += r_dir
                            end_r = start_r + r_offset
                            if end_r >= rows or end_r < 0 or obstacles[end_r, end_c] != 0 \
                                    or owned_pieces[end_r, end_c] != 0:
                                # we hit the edge, one or our own pieces, or an obstacle
                                break

                            if recent_moves[start_r, start_c] == RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value and \
                                    recent_moves[end_r, end_c] == RecentMoves.JUST_CAME_FROM.value and \
                                    enemy_pieces[end_r, end_c] == 0:
                                # piece can't double back here due to rules about oscillating in the same place
                                # doesn't stop the search for moves in this direction,
                                # we just can't go specifically to this space
                                continue

                            # If we reached this line, add this end position as a valid move,
                            move_idx = _get_action_spatial_index_from_positions(
                                rows=rows, columns=columns, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c)

                            valid_moves_mask[move_idx] = 1
                            no_moves_available = False

                            if enemy_pieces[end_r, end_c] != 0:
                                # we're attacking this enemy piece, so we cant move past it
                                break

                    # Then handle horizontal movement cases in the same way as horizontal
                    col_directions = (INT_DTYPE_NP(1), INT_DTYPE_NP(-1))
                    for c_dir in col_directions:
                        c_offset = INT_DTYPE_NP(0)
                        end_r = start_r
                        while True:
                            c_offset += c_dir
                            end_c = start_c + c_offset
                            if end_c >= columns or end_c < 0 or obstacles[end_r, end_c] != 0 \
                                    or owned_pieces[end_r, end_c] != 0:
                                # we hit the edge, one or our own pieces, or an obstacle
                                break

                            if recent_moves[start_r, start_c] == RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value and \
                                    recent_moves[end_r, end_c] == RecentMoves.JUST_CAME_FROM.value and \
                                    enemy_pieces[end_r, end_c] == 0:
                                # piece can't double back here due to rules about oscillating in the same place
                                # doesn't stop the search for moves in this direction,
                                # we just can't go specifically to this space
                                # print("skipping double back move, player ")
                                # print(player)
                                continue

                            # If we reached this line, add this end position as a valid move,
                            move_idx = _get_action_spatial_index_from_positions(
                                rows=rows, columns=columns, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c)

                            valid_moves_mask[move_idx] = 1
                            no_moves_available = False

                            if enemy_pieces[end_r, end_c] != 0:
                                # we're attacking this enemy piece, so we cant move past it
                                break

                else:
                    # the piece type can only move 1 block in each of the four directions
                    for end_r, end_c in [(start_r + 1, start_c), (start_r - 1, start_c), (start_r, start_c + 1),
                                         (start_r, start_c - 1)]:
                        if end_c >= columns or end_r >= rows or end_c < 0 or end_r < 0 or \
                                obstacles[end_r, end_c] != 0 or owned_pieces[end_r, end_c] != 0:
                            # we hit the edge, one or our own pieces, or an obstacle
                            continue

                        if recent_moves[start_r, start_c] == RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value and \
                                recent_moves[end_r, end_c] == RecentMoves.JUST_CAME_FROM.value and \
                                enemy_pieces[end_r, end_c] == 0:
                            # piece can't double back here due to rules about oscillating in the same place
                            continue

                        # If we reached this line, add this end position as a valid move,
                        move_idx = _get_action_spatial_index_from_positions(
                            rows=rows, columns=columns, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c)

                        valid_moves_mask[move_idx] = 1
                        no_moves_available = False

    if no_moves_available:
        valid_moves_mask[0, 0, -1] = 1

    return valid_moves_mask


@jit(INT_DTYPE_JIT[:](INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True,
     fastmath=True, cache=True)
def _get_valid_moves_as_1d_mask(state: np.ndarray, player: INT_DTYPE_NP, action_size: INT_DTYPE_NP,
                                max_possible_actions_per_start_position: INT_DTYPE_NP):
    layers, rows, columns = state.shape

    valid_moves_mask = np.zeros(shape=(np.int64(action_size),), dtype=INT_DTYPE_NP)
    owned_pieces = state[_player_index(player=player)]
    enemy_pieces = state[_player_index(player=-player)]
    obstacles = state[StateLayers.OBSTACLES.value]
    recent_moves = state[_player_moves_index(player=player)]

    no_moves_available = True

    if not state[StateData.GAME_OVER.value]:

        for start_r in range(rows):
            for start_c in range(columns):
                piece_type = owned_pieces[start_r, start_c]

                if piece_type == 0 or piece_type == SP.FLAG.value or piece_type == SP.BOMB.value:
                    continue
                elif piece_type == SP.SCOUT.value:
                    # scouts and move any amount in each of the 4 directions until they hit something
                    # move out in each direction until we hit the edge or a piece/obstacle and incrementally add valid moves

                    # First handle vertical movement cases
                    row_directions = (INT_DTYPE_NP(1), INT_DTYPE_NP(-1))
                    for r_dir in row_directions:
                        r_offset = INT_DTYPE_NP(0)
                        end_c = start_c
                        while True:
                            r_offset += r_dir
                            end_r = start_r + r_offset
                            if end_r < 0 or end_r >= rows or obstacles[end_r, end_c] != 0 or owned_pieces[
                                end_r, end_c] != 0:
                                # we hit the edge, one or our own pieces, or an obstacle
                                break

                            if recent_moves[start_r, start_c] == RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value and \
                                    recent_moves[end_r, end_c] == RecentMoves.JUST_CAME_FROM.value and \
                                    enemy_pieces[end_r, end_c] == 0:
                                # piece can't double back here due to rules about oscillating in the same place
                                # doesn't stop the search for moves in this direction,
                                # we just can't go specifically to this space
                                # print("skipping double back move, player ")
                                # print(player)
                                continue

                            # If we reached this line, add this end position as a valid move,
                            move_idx = _get_action_1d_index_from_positions(
                                rows=rows, columns=columns, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c,
                                max_possible_actions_per_start_position=max_possible_actions_per_start_position)

                            valid_moves_mask[move_idx] = 1
                            no_moves_available = False

                            if enemy_pieces[end_r, end_c] != 0:
                                # we're attacking this enemy piece, so we cant move past it
                                break

                    # Then handle horizontal movement cases in the same way as horizontal
                    col_directions = (INT_DTYPE_NP(1), INT_DTYPE_NP(-1))
                    for c_dir in col_directions:
                        c_offset = INT_DTYPE_NP(0)
                        end_r = start_r
                        while True:
                            c_offset += c_dir
                            end_c = start_c + c_offset
                            if end_c < 0 or end_c >= columns or obstacles[end_r, end_c] != 0 or owned_pieces[
                                end_r, end_c] != 0:
                                # we hit the edge, one or our own pieces, or an obstacle
                                break

                            if recent_moves[start_r, start_c] == RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value and \
                                    recent_moves[end_r, end_c] == RecentMoves.JUST_CAME_FROM.value and \
                                    enemy_pieces[end_r, end_c] == 0:
                                # piece can't double back here due to rules about oscillating in the same place
                                # doesn't stop the search for moves in this direction,
                                # we just can't go specifically to this space
                                # print("skipping double back move, player ")
                                # print(player)
                                continue

                            # If we reached this line, add this end position as a valid move,
                            move_idx = _get_action_1d_index_from_positions(
                                rows=rows, columns=columns, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c,
                                max_possible_actions_per_start_position=max_possible_actions_per_start_position)

                            valid_moves_mask[move_idx] = 1
                            no_moves_available = False

                            if enemy_pieces[end_r, end_c] != 0:
                                # we're attacking this enemy piece, so we cant move past it
                                break

                else:
                    # the piece type can only move 1 block in each of the four directions
                    for end_r, end_c in [(start_r + 1, start_c), (start_r - 1, start_c), (start_r, start_c + 1),
                                         (start_r, start_c - 1)]:
                        if end_c < 0 or end_c >= columns or end_r < 0 or end_r >= rows or obstacles[
                            end_r, end_c] != 0 or owned_pieces[end_r, end_c] != 0:
                            # we hit the edge, one or our own pieces, or an obstacle
                            continue

                        if recent_moves[start_r, start_c] == RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value and \
                                recent_moves[end_r, end_c] == RecentMoves.JUST_CAME_FROM.value and \
                                enemy_pieces[end_r, end_c] == 0:
                            # piece can't double back here due to rules about oscillating in the same place
                            continue

                        # If we reached this line, add this end position as a valid move,
                        move_idx = _get_action_1d_index_from_positions(
                            rows=rows, columns=columns, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c,
                            max_possible_actions_per_start_position=max_possible_actions_per_start_position)

                        valid_moves_mask[move_idx] = 1
                        no_moves_available = False

    if no_moves_available:
        valid_moves_mask[-1] = 1

    return valid_moves_mask


@jit(INT_DTYPE_JIT[:, :, :](INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_state_from_player_perspective(state: np.ndarray, player: INT_DTYPE_NP):
    if player == 1:
        return state
    else:  # player == -1

        flipped_state = state.copy()
        flipped_state[StateLayers.PLAYER_1_PIECES.value] = state[StateLayers.PLAYER_2_PIECES.value, ::-1, ::-1]
        flipped_state[StateLayers.PLAYER_2_PIECES.value] = state[StateLayers.PLAYER_1_PIECES.value, ::-1, ::-1]
        flipped_state[StateLayers.OBSTACLES.value] = state[StateLayers.OBSTACLES.value, ::-1, ::-1]
        flipped_state[StateLayers.PLAYER_1_PO_PIECES.value] = state[StateLayers.PLAYER_2_PO_PIECES.value, ::-1, ::-1]
        flipped_state[StateLayers.PLAYER_2_PO_PIECES.value] = state[StateLayers.PLAYER_1_PO_PIECES.value, ::-1, ::-1]
        flipped_state[StateLayers.PLAYER_1_RECENT_MOVES.value] = state[StateLayers.PLAYER_2_RECENT_MOVES.value, ::-1,
                                                                 ::-1]
        flipped_state[StateLayers.PLAYER_2_RECENT_MOVES.value] = state[StateLayers.PLAYER_1_RECENT_MOVES.value, ::-1,
                                                                 ::-1]
        flipped_state[StateLayers.PLAYER_1_STILL_PIECES.value] = state[StateLayers.PLAYER_2_STILL_PIECES.value, ::-1,
                                                                 ::-1]
        flipped_state[StateLayers.PLAYER_2_STILL_PIECES.value] = state[StateLayers.PLAYER_1_STILL_PIECES.value, ::-1,
                                                                 ::-1]

        # captured piece maps
        p1_cap_start = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value
        p1_cap_end = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value
        p2_cap_start = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value
        p2_cap_end = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value

        flipped_state[p1_cap_start:p1_cap_end] = state[p2_cap_start:p2_cap_end, ::-1, ::-1]
        flipped_state[p2_cap_start:p2_cap_end] = state[p1_cap_start:p1_cap_end, ::-1, ::-1]

        return flipped_state


@jit(types.UniTuple(INT_DTYPE_JIT, 4)(INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT,
                                      INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_action_positions_from_player_perspective(player: INT_DTYPE_NP, start_r: INT_DTYPE_NP, start_c: INT_DTYPE_NP,
                                                  end_r: INT_DTYPE_NP, end_c: INT_DTYPE_NP, rows: INT_DTYPE_NP,
                                                  columns: INT_DTYPE_NP):
    if player == 1:
        return start_r, start_c, end_r, end_c
    else:  # player == -1

        max_r = rows - 1
        max_c = columns - 1

        flipped_start_r = max_r - start_r
        flipped_end_r = max_r - end_r
        flipped_start_c = max_c - start_c
        flipped_end_c = max_c - end_c

        return flipped_start_r, flipped_start_c, flipped_end_r, flipped_end_c


@jit(INT_DTYPE_JIT(INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT),
     nopython=True, fastmath=True, cache=True)
def _get_action_1d_index_from_player_perspective(action_index, player, action_size, rows, columns,
                                                 max_possible_actions_per_start_position):
    if player == 1:
        return action_index
    elif action_index == action_size - 1:
        # action is no-nop
        return action_index
    else:  # player is assumed to be -1

        start_r, start_c, end_r, end_c = _get_action_positions_from_1d_index(rows=rows, columns=columns,
                                                                             action_index=action_index,
                                                                             action_size=action_size,
                                                                             max_possible_actions_per_start_position=max_possible_actions_per_start_position)

        flipped_start_r, flipped_start_c, flipped_end_r, flipped_end_c = _get_action_positions_from_player_perspective(
            player=player, start_r=start_r, start_c=start_c, end_r=end_r, end_c=end_c, rows=rows, columns=columns
        )

        return _get_action_1d_index_from_positions(rows=rows, columns=columns, start_r=flipped_start_r,
                                                   start_c=flipped_start_c, end_r=flipped_end_r, end_c=flipped_end_c,
                                                   max_possible_actions_per_start_position=max_possible_actions_per_start_position)


@jit(
    boolean(INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, boolean),
    nopython=True, fastmath=True, cache=True)
def _is_move_valid_by_position(state: np.ndarray, player: INT_DTYPE_NP, start_r: INT_DTYPE_NP, start_c: INT_DTYPE_NP,
                               end_r: INT_DTYPE_NP, end_c: INT_DTYPE_NP, allow_piece_oscillation: bool):
    layers, rows, columns = state.shape

    owned_pieces = state[_player_index(player=player)]
    enemy_pieces = state[_player_index(player=-player)]
    obstacles = state[StateLayers.OBSTACLES.value]
    recent_moves = state[_player_moves_index(player=player)]

    if state[StateData.GAME_OVER.value]:
        # game is over
        print("game is over")

        return False

    if start_c < 0 or start_c >= columns or start_r < 0 or start_r >= rows or obstacles[start_r, start_c] != 0:
        # out of bounds start position
        print("out of bounds start position")

        return False

    if end_c < 0 or end_c >= columns or end_r < 0 or end_r >= rows or obstacles[end_r, end_c] != 0:
        # out of bounds end position
        print("out of bounds end position")
        return False

    moved_piece_type = owned_pieces[start_r, start_c]
    if moved_piece_type == 0 or moved_piece_type == SP.FLAG.value or moved_piece_type == SP.BOMB.value:
        # No piece at this spot to move or piece here isn't allowed to move
        if enemy_pieces[start_r, start_c] != 0:
            print("tried to move enemy piece")
        else:
            print("No piece at this spot to move or piece here isn't allowed to move")
        return False

    if owned_pieces[end_r, end_c] != 0:
        # moved into owned piece
        print("moved into owned piece")
        return False

    if end_r != start_r and end_c != start_c:
        # moved diagonally
        print("moved diagonally")
        return False

    if recent_moves[start_r, start_c] == RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value and \
            recent_moves[end_r, end_c] == RecentMoves.JUST_CAME_FROM.value and \
            enemy_pieces[end_r, end_c] == 0 and \
            (not allow_piece_oscillation):
        # piece can't double back here due to rules about oscillating in the same place
        print("piece can't double back here due to rules about oscillating in the same place")
        return False

    if moved_piece_type == SP.SCOUT.value:
        # check that there are no other pieces/obstacles between start and end position
        if end_r != start_r:
            vertical_direction = np.sign(end_r - start_r)
            for intermediate_r in range(start_r + vertical_direction, end_r, vertical_direction):
                if owned_pieces[intermediate_r, end_c] != 0 or enemy_pieces[intermediate_r, end_c] != 0 or obstacles[
                    intermediate_r, end_c]:
                    return False
        else:
            horizontal_direction = np.sign(end_c - start_c)
            for intermediate_c in range(start_c + horizontal_direction, end_c, horizontal_direction):
                if owned_pieces[end_r, intermediate_c] != 0 or enemy_pieces[end_r, intermediate_c] != 0 or obstacles[
                    end_r, intermediate_c]:
                    return False
    else:
        if abs(end_r - start_r) > 1 or abs(end_c - start_c) > 1:
            print("Non-scout piece type tried to move more than 1 space in a single turn")
            return False

    return True


@jit(boolean(INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, boolean),
     nopython=True, fastmath=True, cache=True)
def _is_move_valid_by_1d_index(state: np.ndarray, player: INT_DTYPE_NP, action_index: INT_DTYPE_NP,
                               action_size: INT_DTYPE_NP,
                               max_possible_actions_per_start_position: INT_DTYPE_NP, allow_piece_oscillation: bool):
    mpa = max_possible_actions_per_start_position

    layers, rows, columns = state.shape
    if action_index == action_size - 1:
        # action is no-op, so we have to do a full valid actions list calculation
        # to be sure that the player really couldn't do anything else
        valid_actions_mask = _get_valid_moves_as_1d_mask(state=state, player=player, action_size=action_size,
                                                         max_possible_actions_per_start_position=mpa)
        return valid_actions_mask[-1] == 1

    start_r, start_c, end_r, end_c = _get_action_positions_from_1d_index(
        rows=rows, columns=columns, action_index=action_index, action_size=action_size,
        max_possible_actions_per_start_position=mpa)

    result = _is_move_valid_by_position(state=state, player=player, start_r=start_r, start_c=start_c, end_r=end_r,
                                        end_c=end_c, allow_piece_oscillation=allow_piece_oscillation)

    if not result:
        print("action size:")
        print(action_size)
        print("index:")
        print(action_index)
        print("positions:")
        print(start_r, start_c, end_r, end_c)

    return result


@jit(float32(INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT), nopython=True, fastmath=True, cache=True)
def _get_game_ended(state, player):
    if state[StateData.GAME_OVER.value]:
        game_winner = state[StateData.WINNER.value]
        if game_winner == 0.0:
            return 1e-4  # small value for tie
        return game_winner * player  # 1 or -1 depending on whether player or other player was winner
    else:
        return 0.0  # 0 for game isn't over yet


@jit(boolean(INT_DTYPE_JIT[:, :, :]), nopython=True, fastmath=True, cache=True)
def _get_game_result_is_invalid(state):
    if state[StateData.GAME_OVER.value]:
        return bool(state[StateData.ENDING_INVALID.value])
    return False


@jit(FLOAT_DTYPE_JIT(INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, boolean,
                     FLOAT_DTYPE_JIT[:, :]), nopython=True, fastmath=True, cache=True)
def _get_heuristic_rewards_from_move(state: np.ndarray, player: INT_DTYPE_NP, action_index: INT_DTYPE_NP,
                                     action_size: INT_DTYPE_NP,
                                     max_possible_actions_per_start_position: INT_DTYPE_NP,
                                     allow_piece_oscillation: bool, reward_matrix: np.ndarray):
    # if not _is_move_valid_by_1d_index(state=state, player=player, action_index=action_index, action_size=action_size,
    #                                   max_possible_actions_per_start_position=max_possible_actions_per_start_position,
    #                                   allow_piece_oscillation=allow_piece_oscillation):
    #     raise ValueError("Couldn't get heuristic rewards because the move wasn't valid.")
    # past this line, we assume that the action is valid

    layers, rows, columns = state.shape

    if action_index == action_size - 1:
        # Action was no-op
        return 0

    # Gather data to handle game logic
    mpa = _get_max_possible_actions_per_start_position(rows=rows, columns=columns)
    start_r, start_c, end_r, end_c = _get_action_positions_from_1d_index(
        rows=rows, columns=columns, action_index=action_index, action_size=action_size,
        max_possible_actions_per_start_position=mpa)

    owned_pieces = state[_player_index(player=player)]
    enemy_pieces = state[_player_index(player=-player)]

    moved_owned_piece_type = owned_pieces[start_r, start_c]
    dest_enemy_piece_type = enemy_pieces[end_r, end_c]

    reward = reward_matrix[moved_owned_piece_type, dest_enemy_piece_type]

    # print("----")
    # print(moved_owned_piece_type)
    # print("captured")
    # print(dest_enemy_piece_type)
    # print(reward)
    # print("----")

    return reward


@jit(
    INT_DTYPE_JIT[:, :, :](INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT, boolean),
    nopython=True, fastmath=True, cache=True)
def _get_next_state(state: np.ndarray, player: INT_DTYPE_NP, action_index: INT_DTYPE_NP, action_size: INT_DTYPE_NP,
                    max_possible_actions_per_start_position: INT_DTYPE_NP, allow_piece_oscillation: bool):
    if not _is_move_valid_by_1d_index(state=state, player=player, action_index=action_index, action_size=action_size,
                                      max_possible_actions_per_start_position=max_possible_actions_per_start_position,
                                      allow_piece_oscillation=allow_piece_oscillation):
        raise ValueError("Couldn't get the next state because the move wasn't valid.")
    # past this line, we assume that the action is valid

    new_state = state.copy()

    if new_state[StateData.GAME_OVER.value]:
        # state doesn't change after initially ending
        return new_state

    # increment turn count
    new_state[StateData.TURN_COUNT.value] = new_state[StateData.TURN_COUNT.value] + 1

    layers, rows, columns = new_state.shape

    if action_index == action_size - 1:
        # Action was no-op, game ends because player couldn't move. Other player wins the game.
        new_state[StateData.GAME_OVER.value] = 1
        new_state[StateData.WINNER.value] = -player
        return new_state

    # Gather data to handle game logic

    mpa = _get_max_possible_actions_per_start_position(rows=rows, columns=columns)
    start_r, start_c, end_r, end_c = _get_action_positions_from_1d_index(
        rows=rows, columns=columns, action_index=action_index, action_size=action_size,
        max_possible_actions_per_start_position=mpa)

    owned_pieces = new_state[_player_index(player=player)]
    enemy_pieces = new_state[_player_index(player=-player)]

    owned_po_pieces = new_state[_player_po_index(player=player)]
    enemy_po_pieces = new_state[_player_po_index(player=-player)]

    owned_still_pieces = new_state[_player_still_pieces_index(player=player)]
    enemy_still_pieces = new_state[_player_still_pieces_index(player=-player)]

    # Handle still pieces logic
    owned_still_pieces[start_r, start_c] = StillPieces.CANT_BE_FLAG.value
    owned_still_pieces[end_r, end_c] = StillPieces.CANT_BE_FLAG.value
    enemy_still_pieces[end_r, end_c] = StillPieces.CANT_BE_FLAG.value

    moved_owned_piece_type = owned_pieces[start_r, start_c]
    moved_owned_po_piece_type = owned_po_pieces[start_r, start_c]

    dest_enemy_piece_type = enemy_pieces[end_r, end_c]

    # Handle piece move and attack (if moving onto enemy piece)

    owned_pieces[start_r, start_c] = SP.NOPIECE.value  # moved piece just left this spot
    owned_po_pieces[start_r, start_c] = SP.NOPIECE.value

    moved_piece_wins_attack = False
    pieces_tied = False
    if dest_enemy_piece_type == SP.NOPIECE.value:
        # No attack, just normal move

        owned_pieces[end_r, end_c] = moved_owned_piece_type

        if abs(end_r - start_r) > 1 or abs(end_c - start_c) > 1:
            assert moved_owned_piece_type == SP.SCOUT.value
            owned_po_pieces[end_r, end_c] = SP.SCOUT.value  # player's long move distance revealed the piece as a Scout
        else:
            owned_po_pieces[end_r, end_c] = moved_owned_po_piece_type  # player's piece type is not revealed by move

    else:
        # destination spot is contested, calculate winner of attack
        if moved_owned_piece_type == SP.MINER.value and dest_enemy_piece_type == SP.BOMB.value:
            moved_piece_wins_attack = True
        elif moved_owned_piece_type == SP.SPY.value and dest_enemy_piece_type == SP.MARSHALL.value:
            moved_piece_wins_attack = True
        elif dest_enemy_piece_type == SP.FLAG.value:
            # enemy flag has been captured, player wins
            new_state[StateData.GAME_OVER.value] = 1
            new_state[StateData.WINNER.value] = player
            moved_piece_wins_attack = True
        elif dest_enemy_piece_type != SP.BOMB.value:
            # use piece ranks to calculate winner since no special rule applies in this case
            if moved_owned_piece_type == dest_enemy_piece_type:
                pieces_tied = True
            elif moved_owned_piece_type > dest_enemy_piece_type:
                moved_piece_wins_attack = True

        if pieces_tied or moved_piece_wins_attack:
            # Enemy piece destroyed
            enemy_pieces[end_r, end_c] = SP.NOPIECE.value
            enemy_po_pieces[end_r, end_c] = SP.NOPIECE.value

        if moved_piece_wins_attack:
            # Player owned piece moves into destination space
            owned_pieces[end_r, end_c] = moved_owned_piece_type
            owned_po_pieces[end_r, end_c] = moved_owned_piece_type  # player's piece type is revealed by attack

        if not moved_piece_wins_attack and not pieces_tied:
            enemy_po_pieces[end_r, end_c] = dest_enemy_piece_type  # enemy piece won attack and has its type revealed

    # Handle Captured Piece Tracking

    if dest_enemy_piece_type != SP.NOPIECE.value:
        # attack happened
        if not moved_piece_wins_attack:
            # player's moved piece was captured
            owned_captured_piece_layer = new_state[_get_player_captured_piece_layer(player, moved_owned_piece_type)]
            owned_captured_piece_layer[end_r, end_c] = owned_captured_piece_layer[end_r, end_c] + 1

        if moved_piece_wins_attack or pieces_tied:
            # enemy's piece at the destination was captured
            enemy_captured_piece_layer = new_state[_get_player_captured_piece_layer(-player, dest_enemy_piece_type)]
            enemy_captured_piece_layer[end_r, end_c] = enemy_captured_piece_layer[end_r, end_c] + 1

    # Handle Recent Moves Tracking

    old_recent_moves_map = new_state[_player_moves_index(player=player)]
    new_recent_moves_map = np.zeros_like(old_recent_moves_map)

    if dest_enemy_piece_type == SP.NOPIECE.value:
        # an attack didn't happen for this move
        # piece could be oscillating without doing anything so we track its movement in recent moves map
        new_recent_moves_map[start_r, start_c] = RecentMoves.JUST_CAME_FROM.value
        if old_recent_moves_map[end_r, end_c] == RecentMoves.JUST_CAME_FROM.value:
            if old_recent_moves_map[start_r, start_c] == RecentMoves.JUST_ARRIVED_AND_NEXT_DOUBLE_BACK_IS_ILLEGAL.value:
                new_recent_moves_map[end_r, end_c] = RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value
            else:
                new_recent_moves_map[end_r, end_c] = RecentMoves.JUST_ARRIVED_AND_NEXT_DOUBLE_BACK_IS_ILLEGAL.value
        else:
            new_recent_moves_map[end_r, end_c] = RecentMoves.JUST_ARRIVED.value

    new_state[_player_moves_index(player=player)] = new_recent_moves_map

    # If next player cannot make any moves, current player wins.
    new_valid_moves = _get_valid_moves_as_1d_mask(
        state=new_state, player=-player, action_size=action_size,
        max_possible_actions_per_start_position=max_possible_actions_per_start_position)
    if new_valid_moves[-1] == 1:
        new_state[StateData.GAME_OVER.value] = 1
        new_state[StateData.WINNER.value] = player

    # If max_turns was reached and game didn't end on its own,
    # the game is forcefully tied and ending is marked invalid.
    max_turned_count_reached = new_state[StateData.TURN_COUNT.value] >= new_state[StateData.MAX_TURNS.value]
    if max_turned_count_reached and not new_state[StateData.GAME_OVER.value]:
        new_state[StateData.GAME_OVER.value] = 1
        new_state[StateData.ENDING_INVALID.value] = 1

    return new_state


class FullyObservableObsLayers(Enum):
    # Deprecated, use FullyObservableObsLayersExtendedChannels (has individual channels for each piece type)

    PLAYER_1_PIECES = INT_DTYPE_NP(0)
    PLAYER_2_PIECES = INT_DTYPE_NP(1)
    OBSTACLES = INT_DTYPE_NP(2)

    PLAYER_1_RECENT_MOVES = INT_DTYPE_NP(3)
    PLAYER_2_RECENT_MOVES = INT_DTYPE_NP(4)

    PLAYER_1_PO_PIECES = INT_DTYPE_NP(5)
    PLAYER_2_PO_PIECES = INT_DTYPE_NP(6)

    PLAYER_1_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(7)
    PLAYER_1_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(19)
    PLAYER_2_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(19)
    PLAYER_2_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(31)

    PLAYER_1_STILL_PIECES = INT_DTYPE_NP(31)
    PLAYER_2_STILL_PIECES = INT_DTYPE_NP(32)


FULLY_OBSERVABLE_OBS_NUM_LAYERS = INT_DTYPE_NP(33)  # Deprecated, use FULLY_OBSERVABLE_OBS_NUM_LAYERS_EXTENDED


@jit(float32[:, :, :](INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True,
     fastmath=True, cache=True)
def _get_fully_observable_observation(state, player, rows, columns):
    # Deprecated, use _get_fully_observable_observation_extended_channels (has individual channels for each piece type)

    state = _get_state_from_player_perspective(state=state, player=player)

    owned_pieces = state[StateLayers.PLAYER_1_PIECES.value]
    enemy_pieces = state[StateLayers.PLAYER_2_PIECES.value]
    obstacle_map = state[StateLayers.OBSTACLES.value]
    self_recent_moves = state[StateLayers.PLAYER_1_RECENT_MOVES.value]
    enemy_recent_moves = state[StateLayers.PLAYER_2_RECENT_MOVES.value]

    owned_still_pieces = state[StateLayers.PLAYER_1_STILL_PIECES.value]
    enemy_still_pieces = state[StateLayers.PLAYER_2_STILL_PIECES.value]

    owned_po_pieces = state[StateLayers.PLAYER_1_PO_PIECES.value]
    enemy_po_pieces = state[StateLayers.PLAYER_2_PO_PIECES.value]

    # captured piece maps
    p1_cap_start = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value
    p1_cap_end = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value
    p2_cap_start = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value
    p2_cap_end = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value

    observation = np.empty(shape=(np.int64(rows), np.int64(columns), np.int64(FULLY_OBSERVABLE_OBS_NUM_LAYERS)),
                           dtype=np.float32)

    observation[:, :, FullyObservableObsLayers.PLAYER_1_PIECES.value] = owned_pieces
    observation[:, :, FullyObservableObsLayers.PLAYER_2_PIECES.value] = enemy_pieces
    observation[:, :, FullyObservableObsLayers.OBSTACLES.value] = obstacle_map
    observation[:, :, FullyObservableObsLayers.PLAYER_1_RECENT_MOVES.value] = self_recent_moves
    observation[:, :, FullyObservableObsLayers.PLAYER_2_RECENT_MOVES.value] = enemy_recent_moves

    observation[:, :, FullyObservableObsLayers.PLAYER_1_PO_PIECES.value] = owned_po_pieces
    observation[:, :, FullyObservableObsLayers.PLAYER_2_PO_PIECES.value] = enemy_po_pieces

    for state_layer, obs_layer in zip(range(p1_cap_start, p1_cap_end),
                                      range(FullyObservableObsLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value,
                                            FullyObservableObsLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value)):
        observation[:, :, obs_layer] = state[state_layer]

    for state_layer, obs_layer in zip(range(p2_cap_start, p2_cap_end),
                                      range(FullyObservableObsLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value,
                                            FullyObservableObsLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value)):
        observation[:, :, obs_layer] = state[state_layer]

    observation[:, :, FullyObservableObsLayers.PLAYER_1_STILL_PIECES.value] = owned_still_pieces
    observation[:, :, FullyObservableObsLayers.PLAYER_2_STILL_PIECES.value] = enemy_still_pieces

    return observation


class PartiallyObservableObsLayers(Enum):
    # Deprecated, use PartiallyObservableObsLayersExtendedChannels (has individual channels for each piece type)

    PLAYER_1_PIECES = INT_DTYPE_NP(0)  # actual piece values for current player's pieces, can't be SP.UNKNOWN
    PLAYER_1_PO_PIECES = INT_DTYPE_NP(1)  # piece values for current player's pieces, can be SP.UNKNOWN
    PLAYER_2_PO_PIECES = INT_DTYPE_NP(2)  # piece values for opponents pieces, can be SP.UNKNOWN
    OBSTACLES = INT_DTYPE_NP(3)

    PLAYER_1_RECENT_MOVES = INT_DTYPE_NP(
        4)  # only tracks the movement of the last moved piece to prevent oscillating a single piece in place forever
    PLAYER_2_RECENT_MOVES = INT_DTYPE_NP(5)

    PLAYER_1_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(
        6)  # locations and counts of captured pieces, each layer tracks a different piece class
    PLAYER_1_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(18)
    PLAYER_2_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(18)
    PLAYER_2_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(30)

    PLAYER_1_STILL_PIECES = INT_DTYPE_NP(30)
    PLAYER_2_STILL_PIECES = INT_DTYPE_NP(31)


PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS = INT_DTYPE_NP(32)  # Deprecated, use PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS_EXTENDED


@jit(float32[:, :, :](INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True,
     fastmath=True, cache=True)
def _get_partially_observable_observation(state, player, rows, columns):
    # Deprecated, use _get_partially_observable_observation_extended_channels (has individual channels for each piece type)

    state = _get_state_from_player_perspective(state=state, player=player)

    owned_pieces = state[StateLayers.PLAYER_1_PIECES.value]
    owned_po_pieces = state[StateLayers.PLAYER_1_PO_PIECES.value]
    enemy_po_pieces = state[StateLayers.PLAYER_2_PO_PIECES.value]
    obstacle_map = state[StateLayers.OBSTACLES.value]
    self_recent_moves = state[StateLayers.PLAYER_1_RECENT_MOVES.value]
    enemy_recent_moves = state[StateLayers.PLAYER_2_RECENT_MOVES.value]

    owned_still_pieces = state[StateLayers.PLAYER_1_STILL_PIECES.value]
    enemy_still_pieces = state[StateLayers.PLAYER_2_STILL_PIECES.value]

    # captured piece maps
    p1_cap_start = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value
    p1_cap_end = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value
    p2_cap_start = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value
    p2_cap_end = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value

    observation = np.empty(shape=(np.int64(rows), np.int64(columns), np.int64(PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS)),
                           dtype=np.float32)

    observation[:, :, PartiallyObservableObsLayers.PLAYER_1_PIECES.value] = owned_pieces
    observation[:, :, PartiallyObservableObsLayers.PLAYER_1_PO_PIECES.value] = owned_po_pieces
    observation[:, :, PartiallyObservableObsLayers.PLAYER_2_PO_PIECES.value] = enemy_po_pieces
    observation[:, :, PartiallyObservableObsLayers.OBSTACLES.value] = obstacle_map
    observation[:, :, PartiallyObservableObsLayers.PLAYER_1_RECENT_MOVES.value] = self_recent_moves
    observation[:, :, PartiallyObservableObsLayers.PLAYER_2_RECENT_MOVES.value] = enemy_recent_moves

    for state_layer, obs_layer in zip(range(p1_cap_start, p1_cap_end),
                                      range(PartiallyObservableObsLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value,
                                            PartiallyObservableObsLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value)):
        observation[:, :, obs_layer] = state[state_layer]

    for state_layer, obs_layer in zip(range(p2_cap_start, p2_cap_end),
                                      range(PartiallyObservableObsLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value,
                                            PartiallyObservableObsLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value)):
        observation[:, :, obs_layer] = state[state_layer]

    observation[:, :, PartiallyObservableObsLayers.PLAYER_1_STILL_PIECES.value] = owned_still_pieces
    observation[:, :, PartiallyObservableObsLayers.PLAYER_2_STILL_PIECES.value] = enemy_still_pieces

    return observation


class FullyObservableObsLayersExtendedChannels(Enum):
    PLAYER_1_PIECES_RANGE_START = INT_DTYPE_NP(0)  # piece values for current player's pieces, can't be SP.UNKNOWN
    PLAYER_1_PIECES_RANGE_END = INT_DTYPE_NP(12)

    PLAYER_2_PIECES_RANGE_START = INT_DTYPE_NP(12)  # piece values for opponent's pieces, can't be SP.UNKNOWN
    PLAYER_2_PIECES_RANGE_END = INT_DTYPE_NP(24)

    PLAYER_1_PO_PIECES_RANGE_START = INT_DTYPE_NP(24)  # piece values for current player's pieces, can be SP.UNKNOWN
    PLAYER_1_PO_PIECES_RANGE_END = INT_DTYPE_NP(37)

    PLAYER_2_PO_PIECES_RANGE_START = INT_DTYPE_NP(37)  # piece values for opponent's pieces, can be SP.UNKNOWN
    PLAYER_2_PO_PIECES_RANGE_END = INT_DTYPE_NP(50)

    OBSTACLES = INT_DTYPE_NP(50)

    PLAYER_1_RECENT_MOVES = INT_DTYPE_NP(51)
    PLAYER_2_RECENT_MOVES = INT_DTYPE_NP(52)

    PLAYER_1_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(53)
    PLAYER_1_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(65)
    PLAYER_2_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(65)
    PLAYER_2_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(77)

    PLAYER_1_STILL_PIECES = INT_DTYPE_NP(77)
    PLAYER_2_STILL_PIECES = INT_DTYPE_NP(78)


FULLY_OBSERVABLE_OBS_NUM_LAYERS_EXTENDED = INT_DTYPE_NP(79)


@jit(float32[:, :, :](INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True,
     fastmath=True, cache=True)
def _get_fully_observable_observation_extended_channels(state, player, rows, columns):
    state = _get_state_from_player_perspective(state=state, player=player)

    owned_pieces = state[StateLayers.PLAYER_1_PIECES.value]
    enemy_pieces = state[StateLayers.PLAYER_2_PIECES.value]
    obstacle_map = state[StateLayers.OBSTACLES.value]
    self_recent_moves = state[StateLayers.PLAYER_1_RECENT_MOVES.value]
    enemy_recent_moves = state[StateLayers.PLAYER_2_RECENT_MOVES.value]

    owned_still_pieces = state[StateLayers.PLAYER_1_STILL_PIECES.value]
    enemy_still_pieces = state[StateLayers.PLAYER_2_STILL_PIECES.value]

    owned_po_pieces = state[StateLayers.PLAYER_1_PO_PIECES.value]
    enemy_po_pieces = state[StateLayers.PLAYER_2_PO_PIECES.value]

    # captured piece maps
    p1_cap_start = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value
    p1_cap_end = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value
    p2_cap_start = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value
    p2_cap_end = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value

    observation = np.empty(
        shape=(np.int64(rows), np.int64(columns), np.int64(FULLY_OBSERVABLE_OBS_NUM_LAYERS_EXTENDED)), dtype=np.float32)

    # print(owned_pieces.shape)
    # print(enemy_pieces.shape)
    # print(obstacle_map.shape)
    # print(self_recent_moves.shape)
    # print(enemy_recent_moves.shape)

    for piece_type, obs_layer in zip(range(1, 13),
                                     range(FullyObservableObsLayersExtendedChannels.PLAYER_1_PIECES_RANGE_START.value,
                                           FullyObservableObsLayersExtendedChannels.PLAYER_1_PIECES_RANGE_END.value)):
        observation[:, :, obs_layer] = np.asarray(owned_pieces == piece_type, dtype=np.float32)

    for piece_type, obs_layer in zip(range(1, 13),
                                     range(FullyObservableObsLayersExtendedChannels.PLAYER_2_PIECES_RANGE_START.value,
                                           FullyObservableObsLayersExtendedChannels.PLAYER_2_PIECES_RANGE_END.value)):
        observation[:, :, obs_layer] = np.asarray(enemy_pieces == piece_type, dtype=np.float32)

    for piece_type, obs_layer in zip(range(1, 14),
                                     range(
                                         FullyObservableObsLayersExtendedChannels.PLAYER_1_PO_PIECES_RANGE_START.value,
                                         FullyObservableObsLayersExtendedChannels.PLAYER_1_PO_PIECES_RANGE_END.value)):
        observation[:, :, obs_layer] = np.asarray(owned_po_pieces == piece_type, dtype=np.float32)

    for piece_type, obs_layer in zip(range(1, 14),
                                     range(
                                         FullyObservableObsLayersExtendedChannels.PLAYER_2_PO_PIECES_RANGE_START.value,
                                         FullyObservableObsLayersExtendedChannels.PLAYER_2_PO_PIECES_RANGE_END.value)):
        observation[:, :, obs_layer] = np.asarray(enemy_po_pieces == piece_type, dtype=np.float32)

    observation[:, :, FullyObservableObsLayersExtendedChannels.OBSTACLES.value] = obstacle_map
    observation[:, :, FullyObservableObsLayersExtendedChannels.PLAYER_1_RECENT_MOVES.value] = self_recent_moves
    observation[:, :, FullyObservableObsLayersExtendedChannels.PLAYER_2_RECENT_MOVES.value] = enemy_recent_moves

    for state_layer, obs_layer in zip(range(p1_cap_start, p1_cap_end),
                                      range(
                                          FullyObservableObsLayersExtendedChannels.PLAYER_1_CAPTURED_PIECE_RANGE_START.value,
                                          FullyObservableObsLayersExtendedChannels.PLAYER_1_CAPTURED_PIECE_RANGE_END.value)):
        observation[:, :, obs_layer] = state[state_layer]

    for state_layer, obs_layer in zip(range(p2_cap_start, p2_cap_end),
                                      range(
                                          FullyObservableObsLayersExtendedChannels.PLAYER_2_CAPTURED_PIECE_RANGE_START.value,
                                          FullyObservableObsLayersExtendedChannels.PLAYER_2_CAPTURED_PIECE_RANGE_END.value)):
        observation[:, :, obs_layer] = state[state_layer]

    observation[:, :, FullyObservableObsLayersExtendedChannels.PLAYER_1_STILL_PIECES.value] = owned_still_pieces
    observation[:, :, FullyObservableObsLayersExtendedChannels.PLAYER_2_STILL_PIECES.value] = enemy_still_pieces

    return observation


class PartiallyObservableObsLayersExtendedChannels(Enum):
    PLAYER_1_PIECES_RANGE_START = INT_DTYPE_NP(0)  # piece values for current player's pieces, can't be SP.UNKNOWN
    PLAYER_1_PIECES_RANGE_END = INT_DTYPE_NP(12)

    PLAYER_1_PO_PIECES_RANGE_START = INT_DTYPE_NP(12)  # piece values for current player's pieces, can be SP.UNKNOWN
    PLAYER_1_PO_PIECES_RANGE_END = INT_DTYPE_NP(25)

    PLAYER_2_PO_PIECES_RANGE_START = INT_DTYPE_NP(25)  # piece values for opponents pieces, can be SP.UNKNOWN
    PLAYER_2_PO_PIECES_RANGE_END = INT_DTYPE_NP(38)

    OBSTACLES = INT_DTYPE_NP(38)

    PLAYER_1_RECENT_MOVES = INT_DTYPE_NP(
        39)  # only tracks the movement of the last moved piece to prevent oscillating a single piece in place forever
    PLAYER_2_RECENT_MOVES = INT_DTYPE_NP(40)

    PLAYER_1_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(
        41)  # locations and counts of captured pieces, each layer tracks a different piece class
    PLAYER_1_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(53)
    PLAYER_2_CAPTURED_PIECE_RANGE_START = INT_DTYPE_NP(53)
    PLAYER_2_CAPTURED_PIECE_RANGE_END = INT_DTYPE_NP(65)

    PLAYER_1_STILL_PIECES = INT_DTYPE_NP(65)
    PLAYER_2_STILL_PIECES = INT_DTYPE_NP(66)


PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS_EXTENDED = INT_DTYPE_NP(67)


@jit(float32[:, :, :](INT_DTYPE_JIT[:, :, :], INT_DTYPE_JIT, INT_DTYPE_JIT, INT_DTYPE_JIT), nopython=True,
     fastmath=True, cache=True)
def _get_partially_observable_observation_extended_channels(state, player, rows, columns):
    state = _get_state_from_player_perspective(state=state, player=player)

    owned_pieces = state[StateLayers.PLAYER_1_PIECES.value]
    owned_po_pieces = state[StateLayers.PLAYER_1_PO_PIECES.value]
    enemy_po_pieces = state[StateLayers.PLAYER_2_PO_PIECES.value]
    obstacle_map = state[StateLayers.OBSTACLES.value]
    self_recent_moves = state[StateLayers.PLAYER_1_RECENT_MOVES.value]
    enemy_recent_moves = state[StateLayers.PLAYER_2_RECENT_MOVES.value]

    owned_still_pieces = state[StateLayers.PLAYER_1_STILL_PIECES.value]
    enemy_still_pieces = state[StateLayers.PLAYER_2_STILL_PIECES.value]

    # captured piece maps
    p1_cap_start = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value
    p1_cap_end = StateLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value
    p2_cap_start = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value
    p2_cap_end = StateLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value

    observation = np.empty(
        shape=(np.int64(rows), np.int64(columns), np.int64(PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS_EXTENDED)),
        dtype=np.float32)

    for piece_type, obs_layer in zip(range(1, 13),
                                     range(
                                         PartiallyObservableObsLayersExtendedChannels.PLAYER_1_PIECES_RANGE_START.value,
                                         PartiallyObservableObsLayersExtendedChannels.PLAYER_1_PIECES_RANGE_END.value)):
        observation[:, :, obs_layer] = np.asarray(owned_pieces == piece_type, dtype=np.float32)

    for piece_type, obs_layer in zip(range(1, 14),
                                     range(
                                         PartiallyObservableObsLayersExtendedChannels.PLAYER_1_PO_PIECES_RANGE_START.value,
                                         PartiallyObservableObsLayersExtendedChannels.PLAYER_1_PO_PIECES_RANGE_END.value)):
        observation[:, :, obs_layer] = np.asarray(owned_po_pieces == piece_type, dtype=np.float32)

    for piece_type, obs_layer in zip(range(1, 14),
                                     range(
                                         PartiallyObservableObsLayersExtendedChannels.PLAYER_2_PO_PIECES_RANGE_START.value,
                                         PartiallyObservableObsLayersExtendedChannels.PLAYER_2_PO_PIECES_RANGE_END.value)):
        observation[:, :, obs_layer] = np.asarray(enemy_po_pieces == piece_type, dtype=np.float32)

    observation[:, :, PartiallyObservableObsLayersExtendedChannels.OBSTACLES.value] = obstacle_map
    observation[:, :, PartiallyObservableObsLayersExtendedChannels.PLAYER_1_RECENT_MOVES.value] = self_recent_moves
    observation[:, :, PartiallyObservableObsLayersExtendedChannels.PLAYER_2_RECENT_MOVES.value] = enemy_recent_moves

    for state_layer, obs_layer in zip(range(p1_cap_start, p1_cap_end),
                                      range(
                                          PartiallyObservableObsLayersExtendedChannels.PLAYER_1_CAPTURED_PIECE_RANGE_START.value,
                                          PartiallyObservableObsLayersExtendedChannels.PLAYER_1_CAPTURED_PIECE_RANGE_END.value)):
        observation[:, :, obs_layer] = state[state_layer]

    for state_layer, obs_layer in zip(range(p2_cap_start, p2_cap_end),
                                      range(
                                          PartiallyObservableObsLayersExtendedChannels.PLAYER_2_CAPTURED_PIECE_RANGE_START.value,
                                          PartiallyObservableObsLayersExtendedChannels.PLAYER_2_CAPTURED_PIECE_RANGE_END.value)):
        observation[:, :, obs_layer] = state[state_layer]

    observation[:, :, PartiallyObservableObsLayersExtendedChannels.PLAYER_1_STILL_PIECES.value] = owned_still_pieces
    observation[:, :, PartiallyObservableObsLayersExtendedChannels.PLAYER_2_STILL_PIECES.value] = enemy_still_pieces

    return observation


def _get_dict_of_valid_moves_by_position(state: np.ndarray, player, rows, columns, action_size,
                                         max_possible_actions_per_start_position):
    """ Returns dict of valid moves positions where keys are starting positions and values are lists of
    corresponding end positions"""

    # This could be sped up by not making it out of existing functions.

    mpapsp = max_possible_actions_per_start_position

    valid_moves_mask = _get_valid_moves_as_1d_mask(state=state, player=INT_DTYPE_NP(player), action_size=action_size,
                                                   max_possible_actions_per_start_position=mpapsp)

    valid_moves_dict = {}

    for move_idx in range(len(valid_moves_mask)):

        if valid_moves_mask[move_idx]:
            start_r, start_c, end_r, end_c = _get_action_positions_from_1d_index(rows=rows, columns=columns,
                                                                                 action_index=INT_DTYPE_NP(move_idx),
                                                                                 action_size=action_size,
                                                                                 max_possible_actions_per_start_position=mpapsp)

            key = "{},{}".format(start_r, start_c)

            if key not in valid_moves_dict:
                valid_moves_dict[key] = []

            valid_moves_dict[key].append([end_r, end_c])

    return valid_moves_dict
