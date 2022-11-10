"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from enum import Enum, auto


class ControlFactorTypes(Enum):
    NONE = auto()
    CONSTANT = auto()
    LINEAR = auto()
    ADAPTIVE = auto()
