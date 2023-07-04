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



def eliticism_learning_strategy(evolutionary_state : EvolutionaryStates,
                                x : np.array,
                                x_mutation_index : int, gbest_position : np.array,
                                objective_function, domain_boundaries,
                                t_current : int, t_max : int):
    """
    Arguments
    ---------
    - evolutionary_state : EvolutionaryStates

    - x : np.array
        shape = (number of particles, number of dimensions)\n
        A 2D array which represents the positions of the particles of the swarm.

    - domain_boundaries : np.array
        shape = (number of dimensions, 2)\n
        A 2D array whose i-th row contain the lower (position 0) and higher (position 1) bounds of the search domain for the i-th dimension.


    - objective_function : function
        A pointer to a mathematical (scalar) objective function, which expects a 2D array as input.

    """

    # Dynamic sanity checks
    assert(evolutionary_state in EvolutionaryStates)
    assert(x.ndim == 2)

    # This strategy aims at maintaining the diversity during the search,
    # by causing an escape state of the best particle when it gets trapped in a local minimum.
    # That is why this strategy is executed only in case of convergence.
    if evolutionary_state == EvolutionaryStates.CONVERGENCE:

        # Declaring random number generator
        random_generator = np.random.default_rng()

        # Picking a dimension at random for the mutation to take place.
        mutation_dimension = random_generator.integers(x.shape[1])

        domain_dimension_range = domain_boundaries[mutation_dimension, 1] - domain_boundaries[mutation_dimension, 0]
        # "Adaptive Particle Swarm Optimization, Zhi et al." -> "IV. APSO" -> "C. ELS" ->
        # "Empirical study shows that σ_max = 1.0 and σ_min = 0.1
        # result in good performance on most of the test functions"
        sigma = 1 - (1 - 0.1) * (t_current / t_max)
        elitist_learning_rate = random_generator.normal(0, sigma)

        # "The ELS randomly chooses one dimension of gBest’s historical best position [...]" page 9/20, below Fig 7.

        # # ==================== DOES NOT FOLLOW THE PAPER PERFECTLY, BUT IT BEHAVES RIDICULOUSLY GOOD START ====================
        # # 'mutated_position' variable initialization. To be finalized in the following lines of code.
        # mutated_position = x[x_mutation_index] # Creates a view, i.e. `mutated_position` points to particle which at some point had gbest.
        # # Storing the objective value of the best particle before applying the perturbation/mutation

        # previous_objective_value = objective_function(np.expand_dims(mutated_position, axis=0)) # Objective function expects a 2D array.
        # # Applying the perturbation to all dimensions
        # mutated_position += domain_dimension_range * elitist_learning_rate
        # # ==================== DOES NOT FOLLOW THE PAPER PERFECTLY, BUT IT BEHAVES RIDICULOUSLY GOOD START ====================


        # ==================== FAITHFUL TO THE PAPER, BUT DOES NOT BEHAVE AS GOOD START ====================
        # 'mutated_position' variable initialization. To be finalized in the following lines of code.
        mutated_position = gbest_position.copy()
        # Storing the objective value of the best particle before applying the perturbation/mutation

        previous_objective_value = objective_function(np.expand_dims(mutated_position, axis=0)) # Objective function expects a 2D array.
        # Applying the perturbation to the random dimension
        mutated_position[mutation_dimension] += domain_dimension_range * elitist_learning_rate
        # ==================== FAITHFUL TO THE PAPER, BUT DOES NOT BEHAVE AS GOOD START ====================

        # Making sure that the particle will still be inside the search space after the mutation.

        #  Attempt to generate a new random number (always following the same distribution) until a valid one
        #  (one that keeps the particle inside the search space) is generated was made, but experimentally,
        #  it showed that this can slow down the PSO significantly, because of its random nature, especially in the
        #  early stages of the algorithm. Cutting down mutation step range to "search_space_range/2" attempt was made
        #  but it did not lead to a satisfying speed-up.
        
        # With the above information, the "manual" method of clipping the result to the domain was selected in favour of execution speed.
        mutated_position = np.clip(mutated_position, domain_boundaries[mutation_dimension, 0], domain_boundaries[mutation_dimension, 1])


        # If the mutated position achieves a better fitness function, then have the best particle move there.
        # Note: This PSO implementation follows a minimization approach. Therefore, "better" is equivalent to "lower value".
        if objective_function(np.expand_dims(mutated_position, axis=0)) < previous_objective_value:
            x[x_mutation_index] = mutated_position
        else: # Replacing particle with worst position with particle with mutated best position.
            objective_values = objective_function(x)  # NOTE: Recomputing the fitness values may be expensive...maybe store them somewhere in the swarm and pass them to this function?
            x[np.argmax(objective_values)] = mutated_position


    return x
