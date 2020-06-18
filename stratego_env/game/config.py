from stratego_env.game.stratego_procedural_impl import SP

STANDARD_STRATEGO_CONFIG = {
    'rows': 10,
    'columns': 10,
    'max_turns': 2000,
    'obstacle_locations': [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)],
    'piece_amounts': {
        SP.SPY: 1,
        SP.SCOUT: 8,
        SP.MINER: 5,
        SP.SERGEANT: 4,
        SP.LIEUTENANT: 4,
        SP.CAPTAIN: 4,
        SP.MAJOR: 3,
        SP.COLONEL: 2,
        SP.GENERAL: 1,
        SP.MARSHALL: 1,
        SP.FLAG: 1,
        SP.BOMB: 6
    },
    'initial_state_usable_rows': 4
}

MEDIUM_STANDARD_STRATEGO_CONFIG = {
    'rows': 10,
    'columns': 10,
    'max_turns': 800,
    'obstacle_locations': [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)],
    'piece_amounts': {
        SP.SPY: 1,
        SP.SCOUT: 8,
        SP.MINER: 5,
        SP.SERGEANT: 4,
        SP.LIEUTENANT: 4,
        SP.CAPTAIN: 4,
        SP.MAJOR: 3,
        SP.COLONEL: 2,
        SP.GENERAL: 1,
        SP.MARSHALL: 1,
        SP.FLAG: 1,
        SP.BOMB: 6
    },
    'initial_state_usable_rows': 4
}

SHORT_STANDARD_STRATEGO_CONFIG = {
    'rows': 10,
    'columns': 10,
    'max_turns': 400,
    'obstacle_locations': [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)],
    'piece_amounts': {
        SP.SPY: 1,
        SP.SCOUT: 8,
        SP.MINER: 5,
        SP.SERGEANT: 4,
        SP.LIEUTENANT: 4,
        SP.CAPTAIN: 4,
        SP.MAJOR: 3,
        SP.COLONEL: 2,
        SP.GENERAL: 1,
        SP.MARSHALL: 1,
        SP.FLAG: 1,
        SP.BOMB: 6
    },
    'initial_state_usable_rows': 4
}

# WORKS
# STANDARD_STRATEGO_CONFIG2 = {
#     'rows': 10,
#     'columns': 10,
#     'max_turns': 2000,
#     'obstacle_locations': [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)],
#     'piece_amounts': {
#         SP.SPY: 1,
#         SP.SCOUT: 4,
#         SP.MINER: 3,
#         SP.SERGEANT: 3,
#         SP.LIEUTENANT: 3,
#         SP.CAPTAIN: 2,
#         SP.MAJOR: 1,
#         SP.COLONEL: 2,
#         SP.GENERAL: 1,
#         SP.MARSHALL: 1,
#         SP.FLAG: 1,
#         SP.BOMB: 3
#     },
#     'initial_state_usable_rows': 4
# }


# WORKS
# STANDARD_STRATEGO_CONFIG2 = {
#     'rows': 10,
#     'columns': 10,
#     'max_turns': 2000,
#     'obstacle_locations': [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)],
#     'piece_amounts': {
#         SP.SPY: 1,
#         SP.SCOUT: 6,
#         SP.MINER: 4,
#         SP.SERGEANT: 4,
#         SP.LIEUTENANT: 3,
#         SP.CAPTAIN: 3,
#         SP.MAJOR: 2,
#         SP.COLONEL: 2,
#         SP.GENERAL: 1,
#         SP.MARSHALL: 1,
#         SP.FLAG: 1,
#         SP.BOMB: 4
#     },
#     'initial_state_usable_rows': 4
# }


STANDARD_STRATEGO_CONFIG2 = {
    'rows': 15,
    'columns': 15,
    'max_turns': 2000,
    'obstacle_locations': [],
    'piece_amounts': {
        SP.SPY: 0,
        SP.SCOUT: 0,
        SP.MINER: 0,
        SP.SERGEANT: 0,
        SP.LIEUTENANT: 0,
        SP.CAPTAIN: 0,
        SP.MAJOR: 0,
        SP.COLONEL: 3,
        SP.GENERAL: 0,
        SP.MARSHALL: 0,
        SP.FLAG: 1,
        SP.BOMB: 0
    },
    'initial_state_usable_rows': 5
}

# STANDARD_STRATEGO_CONFIG2 = {
#     'rows': 10,
#     'columns': 10,
#     'max_turns': 2000,
#     'obstacle_locations': [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)],
#     'piece_amounts': {
#         SP.SPY: 1,
#         SP.SCOUT: 8,
#         SP.MINER: 5,
#         SP.SERGEANT: 4,
#         SP.LIEUTENANT: 4,
#         SP.CAPTAIN: 4,
#         SP.MAJOR: 3,
#         SP.COLONEL: 2,
#         SP.GENERAL: 1,
#         SP.MARSHALL: 1,
#         SP.FLAG: 1,
#         SP.BOMB: 6
#     },
#     'initial_state_usable_rows': 4
# }

OCTA_BARRAGE_STRATEGO_CONFIG = {
    'rows': 8,
    'columns': 8,
    'max_turns': 1000,
    'obstacle_locations': [(4, 2), (3, 2), (4, 5), (3, 5)],
    'piece_amounts': {
        SP.SPY: 1,
        SP.SCOUT: 2,
        SP.MINER: 1,
        SP.SERGEANT: 0,
        SP.LIEUTENANT: 0,
        SP.CAPTAIN: 0,
        SP.MAJOR: 0,
        SP.COLONEL: 0,
        SP.GENERAL: 1,
        SP.MARSHALL: 1,
        SP.FLAG: 1,
        SP.BOMB: 1
    },
    'initial_state_usable_rows': 3
}

BARRAGE_STRATEGO_CONFIG = {
    'rows': 10,
    'columns': 10,
    'max_turns': 1000,
    'obstacle_locations': [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)],
    'piece_amounts': {
        SP.SPY: 1,
        SP.SCOUT: 2,
        SP.MINER: 1,
        SP.SERGEANT: 0,
        SP.LIEUTENANT: 0,
        SP.CAPTAIN: 0,
        SP.MAJOR: 0,
        SP.COLONEL: 0,
        SP.GENERAL: 1,
        SP.MARSHALL: 1,
        SP.FLAG: 1,
        SP.BOMB: 1
    },
    'initial_state_usable_rows': 4
}

SHORT_BARRAGE_STRATEGO_CONFIG = {
    'rows': 10,
    'columns': 10,
    'max_turns': 100,
    'obstacle_locations': [(4, 2), (5, 2), (4, 3), (5, 3), (4, 6), (5, 6), (4, 7), (5, 7)],
    'piece_amounts': {
        SP.SPY: 1,
        SP.SCOUT: 2,
        SP.MINER: 1,
        SP.SERGEANT: 0,
        SP.LIEUTENANT: 0,
        SP.CAPTAIN: 0,
        SP.MAJOR: 0,
        SP.COLONEL: 0,
        SP.GENERAL: 1,
        SP.MARSHALL: 1,
        SP.FLAG: 1,
        SP.BOMB: 1
    },
    'initial_state_usable_rows': 4
}

MEDIUM_STRATEGO_CONFIG = {
    'rows': 6,
    'columns': 6,
    'max_turns': 200,
    'obstacle_locations': [],
    'piece_amounts': {
        SP.SPY: 0,
        SP.SCOUT: 0,
        SP.MINER: 0,
        SP.SERGEANT: 1,
        SP.LIEUTENANT: 1,
        SP.CAPTAIN: 1,
        SP.MAJOR: 1,
        SP.COLONEL: 1,
        SP.GENERAL: 0,
        SP.MARSHALL: 0,
        SP.FLAG: 1,
        SP.BOMB: 0
    },
    'initial_state_usable_rows': 1
}

FIVES_STRATEGO_CONFIG = {
    'rows': 5,
    'columns': 5,
    'max_turns': 60,
    'obstacle_locations': [],
    'piece_amounts': {
        SP.SPY: 0,
        SP.SCOUT: 0,
        SP.MINER: 0,
        SP.SERGEANT: 1,
        SP.LIEUTENANT: 1,
        SP.CAPTAIN: 1,
        SP.MAJOR: 1,
        SP.COLONEL: 0,
        SP.GENERAL: 0,
        SP.MARSHALL: 0,
        SP.FLAG: 1,
        SP.BOMB: 0
    },
    'initial_state_usable_rows': 1
}

TINY_STRATEGO_CONFIG = {
    'rows': 4,
    'columns': 4,
    'max_turns': 100,
    'obstacle_locations': [],
    'piece_amounts': {
        SP.SPY: 0,
        SP.SCOUT: 0,
        SP.MINER: 0,
        SP.SERGEANT: 0,
        SP.LIEUTENANT: 1,
        SP.CAPTAIN: 1,
        SP.MAJOR: 1,
        SP.COLONEL: 0,
        SP.GENERAL: 0,
        SP.MARSHALL: 0,
        SP.FLAG: 1,
        SP.BOMB: 0
    },
    'initial_state_usable_rows': 1
}

MICRO_STRATEGO_CONFIG = {
    'rows': 3,
    'columns': 4,
    'max_turns': 20,
    'obstacle_locations': [],
    'piece_amounts': {
        SP.SPY: 0,
        SP.SCOUT: 0,
        SP.MINER: 0,
        SP.SERGEANT: 0,
        SP.LIEUTENANT: 1,
        SP.CAPTAIN: 1,
        SP.MAJOR: 0,
        SP.COLONEL: 0,
        SP.GENERAL: 0,
        SP.MARSHALL: 0,
        SP.FLAG: 1,
        SP.BOMB: 0
    },
    'initial_state_usable_rows': 1
}
