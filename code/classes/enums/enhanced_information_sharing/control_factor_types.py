from enum import Enum


class ControlFactorTypes(Enum):
    NONE = 0
    CONSTANT = 1
    LINEAR = 2
    ADAPTIVE = 3
