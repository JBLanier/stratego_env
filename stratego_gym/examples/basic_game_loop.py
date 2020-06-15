from stratego_gym import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions

if __name__ == '__main__':
    config = {
        'version': GameVersions.STANDARD,
        'random_player_assignment': True,
        'human_inits': True,
        'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,
    }

    env = StrategoMultiAgentEnv(env_config=config)

    number_of_games = 1
    for _ in range(number_of_games):
        print("New Game Started")
        obs = env.reset()
        while True:

            assert len(obs.keys()) == 1
            current_player = list(obs.keys())[0]
            assert current_player == 1 or current_player == -1

            board_observation = obs[current_player][ObservationComponents.PARTIAL_OBSERVATION.value]
            valid_actions_mask = obs[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]

            current_player_action = StrategoMultiAgentEnv.sample_random_valid_action(valid_actions_mask)

            obs, rew, done, info = env.step(action_dict={current_player: current_player_action})
            print(f"Player {current_player} made move {current_player_action}")

            if done["__all__"]:
                print(f"Game Finished, player 1 rew: {rew[1]}, player -1 rew: {rew[-1]}")
                break
            else:
                assert all(r == 0.0 for r in rew.values())
