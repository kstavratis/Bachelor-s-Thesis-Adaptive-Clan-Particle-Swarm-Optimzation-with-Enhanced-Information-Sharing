"""
Copyright (C) 2023  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

import numpy as np
from ..enums.evolutionary_states import EvolutionaryStates



def eliticism_learning_strategy(evolutionary_state : EvolutionaryStates, x : np.array, domain_boundaries):

    # Dynamic sanity checks
    assert(evolutionary_state in EvolutionaryStates)
    assert(x.ndim == 2)

    # This strategy aims at maintaining the diversity during the search,
    # by causing a escape state of the best particle when it gets trapped in a local minimum.
    # That is why this strategy is executed only in case of convergence.
    if evolutionary_state == EvolutionaryStates.CONVERGENCE:
        # Picking a dimension at random for the mutation to take place.
        search_space_dimension = np.randint(x.shape[1])

        search_space_range = \
            self._spawn_boundaries[search_space_dimension][1] - self._spawn_boundaries[search_space_dimension][0]
        # "Adaptive Particle Swarm Optimization, Zhi et al." -> "IV. APSO" -> "C. ELS" ->
        # "Empirical study shows that σ_max = 1.0 and σ_min = 0.1
        # result in good performance on most of the test functions"
        sigma = (1 - (1 - 0.1) / self._max_iterations * self.current_iteration)
        elitist_learning_rate = gauss(0, sigma)

        # 'mutated_position' variable initialization. To be finalized in the following lines of code.
        mutated_position = self._particle_with_best_personal_best._position

        # Making sure that the particle will still be inside the search space after the mutation.
        # First two conditions (if and elif) enforce that the particle remains inside the search boundaries where they to be exceeded.
        # Last one (else) applies the calculated transposition as is.
        # TODO Create a better strategy for ensuring that the particle remains in search boundaries.
        #  Attempt to generate a new random number (always following the same distribution) until a valid one
        #  (one that keeps the particle inside the search space) is generated was made, but experimentally,
        #  it showed that this can slow down the PSO significantly, because of its random nature, especially in the
        #  early stages of the algorithm. Cutting down mutation step range to "search_space_range/2" attempt was made
        #  but it did not lead to a satisfying speed-up.
        if self._particle_with_best_personal_best._position[search_space_dimension] + search_space_range * elitist_learning_rate < \
                self._spawn_boundaries[search_space_dimension][0]:
            mutated_position[search_space_dimension] = self._spawn_boundaries[search_space_dimension][0]
        elif self._particle_with_best_personal_best._position[search_space_dimension] + search_space_range * elitist_learning_rate > \
            self._spawn_boundaries[search_space_dimension][1]:
            mutated_position[search_space_dimension] = self._spawn_boundaries[search_space_dimension][1]
        else:
            mutated_position[search_space_dimension] += search_space_range * elitist_learning_rate

        # If the mutated position achieves a better fitness function, then have the best particle move there.
        # Note: This PSO implementation follows a minimization approach. Therefore, "better" is equivalent to "lower value".
        if Particle.fitness_function(mutated_position) < Particle.fitness_function(self._particle_with_best_personal_best._position):
            self._particle_with_best_personal_best._position = mutated_position
        else:  # Replacing particle with worst position with particle with mutated best position.
            current_worst_particle = self._find_particle_with_best_personal_best(greater_than=True)
            self.swarm.remove(current_worst_particle)
            self.swarm.append(Particle(Particle.fitness_function, self._spawn_boundaries, spawn_position=mutated_position))
