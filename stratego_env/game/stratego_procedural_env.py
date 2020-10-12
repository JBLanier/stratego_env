from pickle import dumps
from typing import Tuple

import numpy as np

from stratego_env.game.stratego_procedural_impl import INT_DTYPE_NP, StateLayers, SP, \
    _get_max_possible_actions_per_start_position, _create_initial_state, _get_action_size, \
    _get_action_1d_index_from_positions, _get_action_positions_from_1d_index, _get_valid_moves_as_1d_mask, \
    _get_state_from_player_perspective, _get_action_positions_from_player_perspective, \
    _get_action_1d_index_from_player_perspective, _is_move_valid_by_position, _is_move_valid_by_1d_index, \
    _get_game_ended, _get_game_result_is_invalid, _get_next_state, _get_fully_observable_observation, \
    _get_partially_observable_observation, _get_fully_observable_observation_extended_channels, \
    _get_partially_observable_observation_extended_channels, \
    _get_dict_of_valid_moves_by_position, \
    _get_action_spatial_index_from_positions, _get_action_positions_from_spatial_index, \
    _get_valid_moves_as_spatial_mask, _get_action_1d_index_from_spatial_index, _get_action_spatial_index_from_1d_index, \
    _get_spatial_action_size


class StrategoProceduralEnv(object):
    """StrategoProceduralEnv

        Class that retains a minimal amount of information (rows, columns, action_size, etc.) to make
        function calls simpler to call for a given configuration of the game.
    """

    def __init__(self, rows: int, columns: int):
        if rows < 3 or columns < 3:
            raise ValueError("Both rows and columns have to be at least 3 (you passed rows: {} columns: {})."
                             .format(rows, columns))

        self.rows = INT_DTYPE_NP(rows)
        self.columns = INT_DTYPE_NP(columns)
        self.action_size = _get_action_size(rows=self.rows, columns=self.columns)
        self.spatial_action_size = _get_spatial_action_size(rows=self.rows, columns=self.columns)
        self._mpapsp = _get_max_possible_actions_per_start_position(rows=self.rows, columns=self.columns)

    def create_initial_state(self,
                             obstacle_map: np.ndarray,
                             player_1_initial_piece_map: np.ndarray,
                             player_2_initial_piece_map: np.ndarray,
                             max_turns: int):

        correct_shape = (self.rows, self.columns)
        if not np.array_equal(obstacle_map.shape, correct_shape):
            raise ValueError("obstacle map needs to be of shape {}, was {}", correct_shape,
                             obstacle_map.shape)

        if not np.array_equal(player_1_initial_piece_map.shape, correct_shape):
            raise ValueError("player_1_initial_piece_map map needs to be of shape {}, was {}", correct_shape,
                             player_1_initial_piece_map.shape)

        if not np.array_equal(player_2_initial_piece_map.shape, correct_shape):
            raise ValueError("player_2_initial_piece_map map needs to be of shape {}, was {}", correct_shape,
                             player_2_initial_piece_map.shape)

        return _create_initial_state(obstacle_map=obstacle_map,
                                     player_1_initial_piece_map=player_1_initial_piece_map,
                                     player_2_initial_piece_map=player_2_initial_piece_map,
                                     max_turns=INT_DTYPE_NP(max_turns))

    def get_action_1d_index_from_positions(self, start_r, start_c, end_r, end_c):
        return _get_action_1d_index_from_positions(rows=self.rows, columns=self.columns,
                                                   start_r=INT_DTYPE_NP(start_r), start_c=INT_DTYPE_NP(start_c),
                                                   end_r=INT_DTYPE_NP(end_r), end_c=INT_DTYPE_NP(end_c),
                                                   max_possible_actions_per_start_position=self._mpapsp)

    def get_action_positions_from_1d_index(self, action_index):
        return _get_action_positions_from_1d_index(rows=self.rows, columns=self.columns,
                                                   action_index=INT_DTYPE_NP(action_index),
                                                   action_size=self.action_size,
                                                   max_possible_actions_per_start_position=self._mpapsp)

    def get_valid_moves_as_1d_mask(self, state: np.ndarray, player, player_perspective=False):

        if player_perspective and player == -1:
            state = _get_state_from_player_perspective(state=state, player=player)

        return _get_valid_moves_as_1d_mask(state=state, player=INT_DTYPE_NP(player), action_size=self.action_size,
                                           max_possible_actions_per_start_position=self._mpapsp)

    def get_dict_of_valid_moves_by_position(self, state: np.ndarray, player):
        return _get_dict_of_valid_moves_by_position(state=state, player=player, rows=self.rows, columns=self.columns,
                                                    action_size=self.action_size,
                                                    max_possible_actions_per_start_position=self._mpapsp)

    def is_move_valid_by_position(self, state: np.ndarray, player, start_r, start_c, end_r, end_c,
                                  allow_piece_oscillation=False):
        return _is_move_valid_by_position(state=state, player=INT_DTYPE_NP(player), start_r=INT_DTYPE_NP(start_r),
                                          start_c=INT_DTYPE_NP(start_c), end_r=INT_DTYPE_NP(end_r),
                                          end_c=INT_DTYPE_NP(end_c),
                                          allow_piece_oscillation=allow_piece_oscillation)

    def is_move_valid_by_1d_index(self, state: np.ndarray, player, action_index, allow_piece_oscillation=False):
        return _is_move_valid_by_1d_index(state=state, player=INT_DTYPE_NP(player),
                                          action_index=INT_DTYPE_NP(action_index),
                                          action_size=self.action_size,
                                          max_possible_actions_per_start_position=self._mpapsp,
                                          allow_piece_oscillation=allow_piece_oscillation)

    def get_state_from_player_perspective(self, state: np.ndarray, player):
        return _get_state_from_player_perspective(state=np.asarray(state, dtype=INT_DTYPE_NP),
                                                  player=INT_DTYPE_NP(player))

    def get_action_positions_from_player_perspective(self, player, start_r, start_c, end_r, end_c):
        return _get_action_positions_from_player_perspective(player=player, start_r=start_r, start_c=start_c,
                                                             end_r=end_r, end_c=end_c, rows=self.rows,
                                                             columns=self.columns)

    def get_action_1d_index_from_player_perspective(self, action_index, player):
        return _get_action_1d_index_from_player_perspective(action_index=INT_DTYPE_NP(action_index),
                                                            player=INT_DTYPE_NP(player),
                                                            action_size=self.action_size, rows=self.rows,
                                                            columns=self.columns,
                                                            max_possible_actions_per_start_position=self._mpapsp)

    def get_action_spatial_index_from_positions(self, start_r, start_c, end_r, end_c):
        return _get_action_spatial_index_from_positions(
            rows=self.rows, columns=self.columns,
            start_r=INT_DTYPE_NP(start_r), start_c=INT_DTYPE_NP(start_c),
            end_r=INT_DTYPE_NP(end_r), end_c=INT_DTYPE_NP(end_c))

    def get_action_positions_from_spatial_index(self, spatial_index: np.ndarray):
        return _get_action_positions_from_spatial_index(rows=self.rows, columns=self.columns,
                                                        spatial_index=spatial_index)

    def get_valid_moves_as_spatial_mask(self, state, player):
        return _get_valid_moves_as_spatial_mask(state=state, player=INT_DTYPE_NP(player))

    def get_action_1d_index_from_spatial_index(self, spatial_index):
        return _get_action_1d_index_from_spatial_index(rows=self.rows, columns=self.columns,
                                                       spatial_index=spatial_index,
                                                       max_possible_actions_per_start_position=self._mpapsp)

    def get_action_spatial_index_from_1d_index(self, action_index):
        return _get_action_spatial_index_from_1d_index(rows=self.rows, columns=self.columns,
                                                       action_index=INT_DTYPE_NP(action_index),
                                                       action_size=self.action_size,
                                                       max_possible_actions_per_start_position=self._mpapsp)

    def get_game_ended(self, state: np.ndarray, player):

        return _get_game_ended(state=state, player=INT_DTYPE_NP(player))

    def get_game_result_is_invalid(self, state: np.ndarray):
        return _get_game_result_is_invalid(state)

    def get_next_state(self, state: np.ndarray, player, action_index, allow_piece_oscillation=False):
        new_state = _get_next_state(state=state, player=INT_DTYPE_NP(player), action_index=INT_DTYPE_NP(action_index),
                                    action_size=self.action_size, max_possible_actions_per_start_position=self._mpapsp,
                                    allow_piece_oscillation=allow_piece_oscillation)

        new_player = player * -1

        return new_state, new_player

    def get_fully_observable_observation(self, state: np.ndarray, player):

        return _get_fully_observable_observation(state=state, player=INT_DTYPE_NP(player),
                                                 rows=self.rows, columns=self.columns)

    def get_partially_observable_observation(self, state: np.ndarray, player):
        return _get_partially_observable_observation(state=state, player=INT_DTYPE_NP(player),
                                                     rows=self.rows, columns=self.columns)

    def get_fully_observable_observation_extended_channels(self, state: np.ndarray, player):

        return _get_fully_observable_observation_extended_channels(state=state, player=INT_DTYPE_NP(player),
                                                                   rows=self.rows, columns=self.columns)

    def get_partially_observable_observation_extended_channels(self, state: np.ndarray, player):
        return _get_partially_observable_observation_extended_channels(state=state, player=INT_DTYPE_NP(player),
                                                                       rows=self.rows, columns=self.columns)

    def get_serializable_string_for_fully_observable_state(self, state: np.ndarray):
        return dumps(_get_fully_observable_observation(state=state, player=INT_DTYPE_NP(1), rows=self.rows,
                                                       columns=self.columns))

    def get_serializable_string_for_partially_observable_state(self, state: np.ndarray):
        return dumps(_get_partially_observable_observation(state=state, player=INT_DTYPE_NP(1), rows=self.rows,
                                                           columns=self.columns))

    def print_board_to_console(self, state, partially_observable=False, hide_still_piece_markers=True):

        p2_state_layer = StateLayers.PLAYER_2_PO_PIECES.value if partially_observable else StateLayers.PLAYER_2_PIECES.value

        spaces_per_piece = 4

        layers, rows, columns = state.shape

        width = rows * 6 + 1
        print("    COL", end="")
        for column in range(columns - 1, -1, -1):
            print(f"  {str(column).rjust(2)}  ", end="")
        print("\n       {}".format("-" * width))
        for row in range(rows - 1, -1, -1):
            print(f"Row {str(row).rjust(2)}", end=" ")
            print("|", end="")
            for column in range(columns - 1, -1, -1):
                item = ""
                if state[StateLayers.PLAYER_1_PIECES.value, row, column] != 0:
                    item = str(state[StateLayers.PLAYER_1_PIECES.value, row, column])
                elif state[p2_state_layer, row, column] != 0:
                    item = str(-1 * state[p2_state_layer, row, column])
                elif state[StateLayers.OBSTACLES.value, row, column] != 0:
                    item = "R"

                if not hide_still_piece_markers:
                    if state[StateLayers.PLAYER_1_STILL_PIECES.value, row, column] == 1:
                        item += "a"
                    if state[StateLayers.PLAYER_2_STILL_PIECES.value, row, column] == 1:
                        item += "b"
                print(str(item).rjust(spaces_per_piece), end=" |")
            print("\n       {}".format("-" * width))

#
# if __name__ == '__main__':
#     state = create_initial_state_tiny_stratego()
#     import sys
#     np.set_printoptions(threshold=sys.maxsize)
#
#     print(state)
#
#     display_board(state)
#
#     print("action size: ", _get_action_size(10, 10))
#
#     valid_actions = _get_valid_moves_as_1d_mask(state, player=STATE_DATA_TYPE_NP(1))
#     print(sum(_get_valid_moves_as_1d_mask(state, player=STATE_DATA_TYPE_NP(1))))
#
#     action_index = _get_action_1d_index_from_positions(rows=10
#                                                        ,columns=10,
#                                                        start_r=1,
#                                                        start_c=2,
#                                                        end_r=1,
#                                                        end_c=3,
#                                                        max_possible_actions_per_start_position=_get_max_possible_actions_per_start_position(10,10))
#
#     print(action_index)
#
#     print(_get_action_positions_from_1d_index(10,10, action_index, _get_action_size(10, 10), _get_max_possible_actions_per_start_position(10,10)))
#
#     print(_is_move_valid_by_1d_index(state, player=1, action_index=np.random.choice(list(range(len(valid_actions))),p=valid_actions/sum(valid_actions))))
#
#     print(NUM_STATE_LAYERS)
#
#     new_state = _get_next_state(state = state, player=1, action_index=np.random.choice(list(range(len(valid_actions))),p=valid_actions/sum(valid_actions)))
#
#     display_board(state)
