from stratego_env import StrategoMultiAgentEnv, ObservationComponents, ObservationModes, GameVersions
from stratego_env.examples.util import softmax
import numpy as np


def nnet_choose_action_example(current_player, obs_from_env):
    # observation from the env is dict with multiple components
    board_observation = obs_from_env[current_player][ObservationComponents.PARTIAL_OBSERVATION.value]
    valid_actions_mask = obs_from_env[current_player][ObservationComponents.VALID_ACTIONS_MASK.value]

    # brief example as if we were choosing an action using a neural network.
    nnet_input = board_observation

    # neural network outputs logits in the same shape as the valid_actions_mask (board w x board h x ways_to_move).
    # since all logits here are the same value, this example will output a random valid action
    nnet_example_logits_output = np.ones_like(valid_actions_mask)

    # invalid action logits are changed to be -inf
    invalid_actions_are_neg_inf_valid_are_zero_mask = np.maximum(np.log(valid_actions_mask + 1e-8), np.finfo(np.float32).min)
    filtered_nnet_logits = nnet_example_logits_output + invalid_actions_are_neg_inf_valid_are_zero_mask

    # reshape logits from 3D to 1D since the Stratego env accepts 1D indexes in env.step()
    flattened_filtered_nnet_logits = np.reshape(filtered_nnet_logits, -1)

    # get action probabilities using a softmax over the filtered network logit outputs
    action_probabilities = softmax(flattened_filtered_nnet_logits)

    # choose an action from the output probabilities
    chosen_action_index = np.random.choice(range(len(flattened_filtered_nnet_logits)), p=action_probabilities)

    return chosen_action_index


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

            current_player_action = nnet_choose_action_example(current_player=current_player, obs_from_env=obs)

            obs, rew, done, info = env.step(action_dict={current_player: current_player_action})
            print(f"Player {current_player} made move {current_player_action}")

            if done["__all__"]:
                print(f"Game Finished, player 1 rew: {rew[1]}, player -1 rew: {rew[-1]}")
                break
            else:
                assert all(r == 0.0 for r in rew.values())
