"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from enum import Enum


class WallTypes(Enum):
    NONE = 0
    ELIMINATING = 1
    ABSORBING = 2
    REFLECTING = 3
    INVISIBLE = 4
