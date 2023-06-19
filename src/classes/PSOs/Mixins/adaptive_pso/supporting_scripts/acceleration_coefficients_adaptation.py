"""
Copyright (C) 2023  Konstantinos Stavratis
e-mail: kostauratis@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from ..enums.evolutionary_states import EvolutionaryStates
from ..enums.acceleration_coefficients_adaptation_operations import CoefficientOperations


def determine_acceleration_coefficients(evolutionary_state : EvolutionaryStates, c1 : float, c2 : float, c_min : float, c_max : float) -> float:

    # Dynamic sanity checks.
    assert(evolutionary_state in EvolutionaryStates)
    assert(c_min <= c1 <= c_max and c_min <= c2 <= c_max)


    c1_strategy, c2_strategy = "", "" # Declare variables
    if evolutionary_state == EvolutionaryStates.EXPLORATION:
        c1_strategy = CoefficientOperations.INCREASE
        c2_strategy = CoefficientOperations.DECREASE
    elif evolutionary_state == EvolutionaryStates.EXPLOITATION:
        c1_strategy = CoefficientOperations.INCREASE_SLIGHTLY
        c2_strategy = CoefficientOperations.DECREASE_SLIGHTLY
    elif evolutionary_state == EvolutionaryStates.CONVERGENCE:
        c1_strategy = CoefficientOperations.INCREASE_SLIGHTLY
        c2_strategy = CoefficientOperations.INCREASE_SLIGHTLY
    elif evolutionary_state == EvolutionaryStates.JUMP_OUT:
        c1_strategy = CoefficientOperations.DECREASE
        c2_strategy = CoefficientOperations.INCREASE
    else:
        raise ValueError(f"The evolutionary state can only be in one of the {len(EvolutionaryStates)} allowed states {[state for state in EvolutionaryStates.__members__.keys()]}.\n\
        The 'evolutionary_state' parameter has the value {evolutionary_state} instead.")


    # Declaring random number generator
    random_generator = np.random.default_rng()
    # Math: |c_i^{t+1} - c_i^{t}| \leq \delta, \text{where } \delta \text{ is called the "acceleration_rate"}
    # "Experiments reveal that a uniformly generated random value of δ in the interval [0.05, 0.1] performs best on most of the test functions".
    # "Note that we use 0.5 δ in strategies 2 and 3, where “slight” changes are recommended"
    acceleration_rate = random_generator.choice(np.array([random_generator.uniform(0.05, 0.10), random_generator.uniform(-0.10, -0.05)]), 1).item()

    if c1_strategy == CoefficientOperations.INCREASE:
        if c1 + acceleration_rate <= c_max:
            c1 += acceleration_rate
    elif c1_strategy == CoefficientOperations.INCREASE_SLIGHTLY:
        if c1 + 0.5 * acceleration_rate <= c_max:
            c1 += 0.5 * acceleration_rate
    elif c1_strategy == CoefficientOperations.DECREASE:
        if c1 - acceleration_rate >= c_min:
            c1 -= acceleration_rate

    if c2_strategy == CoefficientOperations.INCREASE:
        if c2 + acceleration_rate <= c_max:
            c2 += acceleration_rate
    elif c2_strategy == CoefficientOperations.INCREASE_SLIGHTLY:
        if c2 + 0.5 * acceleration_rate <= c_max:
            c2 += 0.5 * acceleration_rate
    elif c2_strategy == CoefficientOperations.DECREASE:
        if c2 - acceleration_rate >= c_min:
            c2 -= acceleration_rate
    elif c2_strategy == CoefficientOperations.DECREASE_SLIGHTLY:
        if c2 - 0.5 * acceleration_rate >= c_min:
            c2 -= 0.5 * acceleration_rate

    
    # Both acceleration coefficients, c1 and c2, are clamped by the interval [c_min, c_max]
    c1 = np.clip(c1, c_min, c_max) ; c2 = np.clip(c2, c_min, c_max)

    # In order to avoid an explosion state, it is necessary to bound the sum c1+c2.
    # As the minimum and the maximum value for c1 and c2 are c_min = 1.5 and c_max = 2.5,
    # when c1 + c2 > cm + cmax, one should update c1 and c2 using:
    # c_i = c_i/(c1 + c2) (c_min + c_max):
    if c1 + c2 > c_min + c_max:
        c1_old, c2_old = c1, c2
        c1 = c1_old / (c1_old + c2_old) * (c_min + c_max)
        c2 = c2_old / (c1_old + c2_old) * (c_min + c_max)


    # Dynamic sanity checks.
    assert(c_min <= c1 <= c_max and c_min <= c2 <= c_max)

    return c1, c2
