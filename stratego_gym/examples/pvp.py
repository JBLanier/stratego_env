from stratego_gym.game.enums import GameVersions
from stratego_gym.game.stratego_human_server import StrategoHumanGUIServer
from stratego_gym.game.stratego_procedural_env import StrategoProceduralEnv
from stratego_gym.game.util import get_random_human_init_fn
from stratego_gym.stratego_multiagent_env import VERSION_CONFIGS

"""
Launches a game UI at 
http://localhost:7000?player=1   and
http://localhost:7000?player=2

add &po=False to see -all- board pieces
(e.g. http://localhost:7000?player=1&po=False )
"""

if __name__ == '__main__':

    version = GameVersions.STANDARD
    version_config = VERSION_CONFIGS[version]

    base_env = StrategoProceduralEnv(version_config['rows'], version_config['columns'])

    port = 7000
    s = StrategoHumanGUIServer(base_env=base_env, port=port)

    print("Visit \n"
          "http://localhost:{}?player=1 or \n"
          "http://localhost:{}?player=2 on a web browser".format(port, port))

    while True:
        print("waiting for action now")
        player = 1
        # random_initial_state_fn = get_random_initial_state_fn(base_env=base_env, game_version_config=version_config)

        random_initial_state_fn = get_random_human_init_fn(game_version=version, game_version_config=version_config,
                                                           procedural_env=base_env)

        state = random_initial_state_fn()

        s.reset_game(initial_state=state)

        while base_env.get_game_ended(state, player) == 0:
            """" Uncomment this to see the game in console """
            # base_env.print_fully_observable_board_to_console(state)

            action = s.get_action_by_position(state=state, player=player)
            print("action received by client is ", action)
            action_index = base_env.get_action_1d_index_from_positions(*action)
            action_index = base_env.get_action_1d_index_from_player_perspective(action_index=action_index,
                                                                                player=player)
            print("action_index: ", action_index)
            print("action is valid: ", base_env.is_move_valid_by_1d_index(
                state, player, base_env.get_action_1d_index_from_player_perspective(
                    action_index=action_index, player=player)))

            state, player = base_env.get_next_state(state=state, player=player, action_index=action_index)
