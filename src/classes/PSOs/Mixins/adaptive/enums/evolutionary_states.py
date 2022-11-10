"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from enum import Enum


class EvolutionaryStates(Enum):
    EXPLORATION = 0
    EXPLOITATION = 1
    CONVERGENCE = 2
    JUMP_OUT = 3
