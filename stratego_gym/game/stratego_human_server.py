import os
import threading
import time
from multiprocessing import Process

import eventlet
import numpy as np
import socketio
from socketio.exceptions import ConnectionError

from stratego_gym.game.config import STANDARD_STRATEGO_CONFIG
from stratego_gym.game.stratego_procedural_env import StrategoProceduralEnv

STATIC_FILES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "public")

PLAYER_1_ROOM = "/player1"
PLAYER_2_ROOM = "/player2"


def _player_room(player: int):
    if player == 1:
        return PLAYER_1_ROOM
    elif player == -1:
        return PLAYER_2_ROOM
    else:
        raise ValueError("There isn't a namespace for player {}".format(player))


def gui_websocket_broker(base_env: StrategoProceduralEnv, port: int):
    sio = socketio.Server()

    static_files = {
        '/': {'content_type': 'text/html', 'filename': os.path.join(STATIC_FILES_PATH, 'index.html')},
        '/static': STATIC_FILES_PATH
    }

    app = socketio.WSGIApp(sio, static_files=static_files)

    current_state = {'current_action_step': -1}

    @sio.event
    def connect(sid, environ):
        print('connect ', sid)

    @sio.event
    def join_both_player_rooms(sid):
        print("{} entered both player rooms".format(sid))
        sio.enter_room(sid=sid, room=PLAYER_1_ROOM)
        sio.enter_room(sid=sid, room=PLAYER_2_ROOM)

    @sio.event
    def join_player_room(sid, player):
        print("{} entered room {}".format(sid, _player_room(player)))
        sio.enter_room(sid=sid, room=_player_room(player))
        if player in current_state:
            print("emitted state update")
            sio.emit("state_update_wait_for_action_request",
                     data=(current_state[player], current_state['current_action_step'], False),
                     room=_player_room(player))

    @sio.event
    def action_requested_from_env(sid, state, player, action_step, valid_moves):
        # print("server: action requested, player {}".format(player))
        sio.emit("action_requested", data=(state, action_step, valid_moves), room=_player_room(player))

        if current_state['current_action_step'] < action_step:
            current_state['current_action_step'] = action_step
            current_state[player] = state
            current_state[-player] = base_env.get_state_from_player_perspective(state=state, player=-1).tolist()
            sio.emit("state_update_wait_for_action_request",
                     data=(current_state[-player], current_state['current_action_step'], False),
                     room=_player_room(-player))

    @sio.event
    def reset_game(sid, initial_state):
        current_state['current_action_step'] = 0
        current_state[1] = initial_state
        current_state[-1] = base_env.get_state_from_player_perspective(state=initial_state, player=-1).tolist()

        sio.emit("state_update_wait_for_action_request",
                 data=(current_state[1], current_state['current_action_step'], True), room=_player_room(1))

        sio.emit("state_update_wait_for_action_request",
                 data=(current_state[-1], current_state['current_action_step'], True), room=_player_room(-1))

    @sio.event
    def action_selected_by_browser(sid, action_step, action_positions, player):
        print("server got action selected")
        sio.emit("action_selected_for_env", (action_step, action_positions, player))

    @sio.event
    def disconnect(sid):
        print('disconnect ', sid)

    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', port)), app, log_output=False)


class StrategoHumanGUIServer(object):

    def __init__(self, base_env: StrategoProceduralEnv, port: int = 7000):
        self.state = None
        self.base_env: StrategoProceduralEnv = base_env

        self.broker_p = Process(target=gui_websocket_broker, args=(self.base_env, port), daemon=True)
        self.broker_p.start()

        self.sio = socketio.Client(logger=False)
        self.action_step_lock = threading.Lock()
        self.current_action_step_num = 0
        self.current_action_player = None
        self.current_action = None

        @self.sio.event
        def connect():
            print('connection established')
            self.sio.emit("join_both_player_rooms")

        @self.sio.event
        def action_selected_for_env(action_step, action, player):
            print('action {}, received with step {} from player {}'.format(action, action_step, player))

            with self.action_step_lock:
                if action_step == self.current_action_step_num and player == self.current_action_player:
                    self.current_action = action

        @self.sio.event
        def disconnect():
            print('disconnected from server')

        retry_secs = 0.1
        retries = 5
        for retry in range(retries):
            try:
                self.sio.connect('http://localhost:{}'.format(port), )
                break
            except ConnectionError:

                if retry + 1 >= retries:
                    raise ConnectionError

                time.sleep(retry_secs)
                retry_secs *= 2

    def reset_game(self, initial_state: np.ndarray):
        with self.action_step_lock:
            self.current_action_step_num = 0
            self.current_action_player = None

        self.sio.emit(event="reset_game", data=initial_state.tolist())

        self.state = initial_state

    def get_action_by_position(self, state: np.ndarray, player):
        action = None

        player_perspective_state = self.base_env.get_state_from_player_perspective(state=state, player=player)
        pp_valid_moves = self.base_env.get_dict_of_valid_moves_by_position(state=player_perspective_state, player=1)

        with self.action_step_lock:
            self.current_action_player = player

        print("waiting for action from player", player)
        while action is None:
            with self.action_step_lock:
                self.sio.emit(event="action_requested_from_env",
                              data=(player_perspective_state.tolist(),
                                    player,
                                    self.current_action_step_num,
                                    pp_valid_moves))

                if self.current_action is not None:
                    action = self.current_action
                    self.current_action = None
                    self.current_action_step_num += 1

            if action is None:
                time.sleep(0.1)

        return action

    def __del__(self):
        self.broker_p.kill()


if __name__ == '__main__':
    config = STANDARD_STRATEGO_CONFIG

    base_env = StrategoProceduralEnv(config['rows'], config['columns'])

    s = StrategoHumanGUIServer(base_env=base_env)

    # while True:
    #     print("waiting for action now")
    #     player = 1
    #     random_initial_state_fn = get_random_initial_state_fn(base_env=base_env, game_version_config=config)
    #     state = random_initial_state_fn()
    #     base_env.print_fully_observable_board_to_console(state)
    #
    #     s.reset_game(initial_state=state)
    #
    #     while base_env.get_game_ended(state, player) == 0:
    #         # if player == 1:
    #         #     base_env.print_fully_observable_board_to_console(state)
    #         #     action = s.get_action_by_position(state=state, player=player)
    #         #     print("action received by client is ", action)
    #         #     action_index = base_env.get_action_1d_index_from_positions(*action)
    #         #     action_index = base_env.get_action_1d_index_from_player_perspective(action_index=action_index, player=player)
    #         #     print("action_size:", base_env.action_size)
    #         #     print("action_index: ", action_index)
    #         #     print("action is valid: ", base_env.is_move_valid_by_1d_index(state, player, base_env.get_action_1d_index_from_player_perspective(action_index=action_index, player=player)))
    #         # else:
    #
    #         action_index = base_env.sample_partially_observable_simple_heuristic_policy(state=state, player=player)
    #         time.sleep(0.1)
    #
    #         state, player = base_env.get_next_state(state=state, player=player, action_index=action_index)
    #         s.reset_game(state)
    #         base_env.print_fully_observable_board_to_console(state)
