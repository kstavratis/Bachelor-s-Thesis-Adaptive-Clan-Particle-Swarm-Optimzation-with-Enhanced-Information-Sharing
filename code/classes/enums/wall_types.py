from enum import Enum


class WallTypes(Enum):
    NONE = 0
    ELIMINATING = 1
    ABSORBING = 2
    REFLECTING = 3
    INVISIBLE = 4
