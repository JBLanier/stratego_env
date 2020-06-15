from stratego_gym import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions

if __name__ == '__main__':
    config = {
        'version': GameVersions.STANDARD,
        'random_player_assignment': False,
        'human_inits': True,
        'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,

        'vs_human': True,  # one of the players is a human using a web gui
        'human_player_num': -1,  # 1 or -1
        'human_web_gui_port': 7000,
    }

    env = StrategoMultiAgentEnv(env_config=config)

    print(f"Visit \nhttp://localhost:{config['human_web_gui_port']}?player={config['human_player_num']} on a web browser")
    env_agent_player_num = config['human_player_num'] * -1

    number_of_games = 2
    for _ in range(number_of_games):
        print("New Game Started")
        obs = env.reset()
        while True:

            assert len(obs.keys()) == 1
            current_player = list(obs.keys())[0]
            assert current_player == env_agent_player_num

            board_observation = obs[current_player][ObservationComponents.PARTIAL_OBSERVATION.value]
            valid_actions_mask = obs[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]

            current_player_action = StrategoMultiAgentEnv.sample_random_valid_action(valid_actions_mask)

            obs, rew, done, info = env.step(action_dict={current_player: current_player_action})
            print(f"Player {current_player} made move {current_player_action}")

            if done["__all__"]:
                print(f"Game Finished, player {env_agent_player_num} rew: {rew[env_agent_player_num]}")
                break
            else:
                assert all(r == 0.0 for r in rew.values())
