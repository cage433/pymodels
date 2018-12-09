from enum import Enum

class OptionRight(Enum):
    CALL=1
    PUT=2
    STRADDLE=3

class ExerciseStyle(Enum):
    EUROPEAN=1,
    AMERICAN=2

