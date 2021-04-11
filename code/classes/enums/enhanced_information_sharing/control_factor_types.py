"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from enum import Enum


class ControlFactorTypes(Enum):
    NONE = 0
    CONSTANT = 1
    LINEAR = 2
    ADAPTIVE = 3
