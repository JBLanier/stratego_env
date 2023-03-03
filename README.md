# Stratego Env
Multi-Agent RL Environment for the Stratego Board Game (and variants)

Stratego Env provides a gym-like interface for the 2-player, zero-sum, imperfect-information game, Stratego, its smaller variants like Barrage, and toy-sized versions of the game.

# Installation
Tested with python 3.6  
(cd into this repo's directory)
```
pip install -e .
```

# Usage
Included in the [examples](https://github.com/JBLanier/stratego_env/tree/master/stratego_env/examples) folder are:

[basic_game_loop.py](https://github.com/JBLanier/stratego_env/blob/master/stratego_env/examples/basic_game_loop.py)  
A basic loop demonstrating RL env functionality with multiple agents.

[env_agent_vs_human.py](https://github.com/JBLanier/stratego_env/blob/master/stratego_env/examples/env_agent_vs_human.py)  
A similar loop as above, demonstrating RL env functionality for a single agent against a human using a web-browser interface.

[pvp.py](https://github.com/JBLanier/stratego_env/blob/master/stratego_env/examples/pvp.py)  
A loop for two humans to player each other using a web-browser interface.

## Valid Actions and Action Space Shapes

In Stratego, only certain actions are valid/legal in each turn. In our implementation, choosing an invalid action throws an exception.
To compensate for this, the observation space is a dict comprising of multiple parts, including the actual "board" view and a mask of valid actions, where 1 means valid and 0 means invalid.

In our deep reinforcement learning experiments using convolutional networks, a spatially oriented 3D discrete action space was easier to learn with compared to a 1D discrete action space.  

The valid_actions_mask observation component is of the shape, (board_width x board_height x ways_to_move_a_peice).  

To maintain compatibility with existing RL libraries, our environment still accepts a 1D action space for env.step(). Even though the valid_actions_mask observation component is 3D, you will have to convert any chosen actions in 3D space to 1D space in order to input it to env.step().

This concept can best be shown through this example as if we were to use a neural network policy to choose an action:

```python
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
    invalid_actions_are_neg_inf_valid_are_zero_mask = np.maximum(np.log(valid_actions_mask), np.finfo(np.float32).min)
    filtered_nnet_logits = nnet_example_logits_output + invalid_actions_are_neg_inf_valid_are_zero_mask

    # reshape logits from 3D to 1D since the Stratego env accepts 1D indexes in env.step()
    flattened_filtered_nnet_logits = np.reshape(filtered_nnet_logits, -1)

    # get action probabilities using a softmax over the filtered network logit outputs
    action_probabilities = softmax(flattened_filtered_nnet_logits)

    # choose an action from the output probabilities
    chosen_action_index = np.random.choice(range(len(flattened_filtered_nnet_logits)), p=action_probabilities)

    return chosen_action_index
```
This function can be seen in context in [basic_game_loop.py](https://github.com/JBLanier/stratego_env/blob/master/stratego_env/examples/basic_game_loop.py)
and [env_agent_vs_human.py](https://github.com/JBLanier/stratego_env/blob/master/stratego_env/examples/env_agent_vs_human.py)

## Initial Piece Starting Positions

Our Stratego Environment currently does not provide an RL env interface for providing starting piece positions. Our current focus is on performing well after this step is completed. For Stratego and Barrage, each player is provided at the start with random setups sampled from human games from the [Gravon Stratego archive](https://www.gravon.de/gravon/stratego/stratego.jsp). For other toy variants, each player is provided with random setups.
