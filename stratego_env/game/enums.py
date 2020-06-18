from enum import Enum


class ObservationModes(Enum):
    PARTIALLY_OBSERVABLE = 'partially_observable'
    FULLY_OBSERVABLE = 'fully_observable'
    BOTH_OBSERVATIONS = 'both_observations'


# keys in each observation returned by the environment
class ObservationComponents(Enum):
    PARTIAL_OBSERVATION = 'partial_observation'
    FULL_OBSERVATION = 'full_observation'
    VALID_ACTIONS_MASK = 'valid_actions_mask'
    INTERNAL_STATE = 'internal_state'


class GameVersions(Enum):
    STANDARD = 'standard'
    SHORT_STANDARD = 'short_standard'
    MEDIUM_STANDARD = 'medium_standard'
    STANDARD2 = 'standard2'
    BARRAGE = 'barrage'
    SHORT_BARRAGE = 'short_barrage'
    OCTA_BARRAGE = 'octa_barrage'
    MEDIUM = 'medium'
    TINY = 'tiny'
    MICRO = 'micro'
    FIVES = 'fives'
