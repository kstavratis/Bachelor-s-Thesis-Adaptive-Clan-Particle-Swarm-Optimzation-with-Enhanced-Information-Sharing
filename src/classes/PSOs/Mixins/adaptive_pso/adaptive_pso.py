"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

import numpy as np

import numbers
from random import uniform, gauss, randrange

from .supporting_scripts.evolutionary_state_classification_singleton_method import classify_evolutionary_state
from .supporting_scripts.acceleration_coefficients_adaptation import determine_acceleration_coefficients
from .enums.evolutionary_states import EvolutionaryStates


class AdaptivePSO:
    """
    The Adaptive Particle Swarm Optimization (APSO) as described in the paper:
    "Zhan, Zhi-Hui & Zhang, Jun & Li, Yun & Chung, Henry. (2010).
    Adaptive Particle Swarm Optimization.
    Systems, Man, and Cybernetics, Part B: Cybernetics, IEEE Transactions on. 39. 1362 - 1381. 10.1109/TSMCB.2009.2015956.
    "

    As a short overview of the paper, the adaptivity of the swarm is done as follows:
    STEP 1: Compute the evolutionary factor of the swarm.
    STEP 2: Classify the swarm into one of the four states: "Exploration", "Exploitation", "Convergence", "Jump-Out".
            and adapt the learning factors according the class. 
    STEP 2b: Introduce a small energy (perturbation) into the system in case it is in the "Convergence" state.
    STEP 3: Update the inertia weight ω as a function of the evolutionary factor # Math: \omega = \frac{1}{1 + 1.5\exp{(-2.6 \cdot f_{evol})}} 
    
    Attributes
    ----------
    c_min : float
        Lower bound below which the c1 and c2 learning factors of the swarm are clampled
        The value the paper proposes is 1.5
    
    c_max : float
        Upper bound after which the c1 and c2 learning factors of the swarm are clamped.
        The value the paper proposes is 2.5
    """
    def __init__(self, c_min : float, c_max : float):
        self.__c_min, self.__c_max, self.__c_min_max_sum = c_min, c_max, c_min + c_max
        self.__evolutionary_state = None # Declare variable

    def update_weights_and_acceleration_coefficients(self) -> None:
        # The evolutionary factor, f_evol, is required to update both the inertia weight --with the use of a closed-form formula--
        # and the acceleration coefficients c1 & c2.
        f_evol = self.__compute_evolutionary_factor(self.__average_between_each_particle_to_all_others())

        # Adapt inertia weight
        self.w = self._adapt_inertia_factor(f_evol)

        # Adapt acceleration_coefficients
        evolutionary_state = self._classify_evolutionary_state(f_evol)
        self.c1, self.c2 = determine_acceleration_coefficients(evolutionary_state, self.c1, self.c2, self.__c_min, self.__c_max)

    def _classify_evolutionary_state(self, f_evol):
        # Sanity check
        if (isinstance(f_evol, numbers.Real)):
            return classify_evolutionary_state(f_evol)
        else:
            raise TypeError(f'Expected type of parameter evolutionary factor f_evol to be rational number {type(numbers.Real)}.\n'\
            f'{type(f_evol)} was provided instead.')
        
    def _estimate_evolutionary_state(self):

        return self.__compute_evolutionary_factor(self.__average_between_each_particle_to_all_others())
    
    def _adapt_inertia_factor(self, f_evol: float):
        self.w = 1 / (1 + 1.5 * np.exp(-2.6 * f_evol))  # ∈[0.4, 0.9]  ∀f_evol ∈[0,1]
    

    def __compute_evolutionary_factor(self, d : np.array) -> float:
        """
        Parameters
        ----------
        d : np.array
            - shape = (n, ), where n is the number of rows of "self.swarm_positions",
            which in turn encodes the number of particles in the swarm.

            - d[i] := The average (euclidean) distance of particle i with all other particles j: # Math: d_i = \frac{1}{N-1} \sum_{i=1}^{N} \| \mathbf{x}_i - \mathbf{x}_j \| 

        Returns
        -------
        : float
            The evolutionary factor: # Math: f_{evol} = \frac{d_g - d_{min}}{d_{max} - d_{min}}
        """

        index_of_globablly_best_particle = np.where(np.all(self.pbest_positions == self.gbest_position, axis=1))[0].item()
        # NOTE: Explanation as to why the above command makes sense.
        # Quoting the paper "Denote di of the globally best particle dg".
        # A question that may arise is "how is the globally best particle known at all times?"
        # There are only two cases:
        #   a) The paper refers to the particle which at some point had the global position.
        #   b) The paper refers to the particle which has the best objective value at the current iteration.
        #   There are cases where a) and b) are not the same particle.
        #   This may happen when the particles "shoot over" better solutions, "landing" on only worse than the currently global best.
        # This implementation assumes case a), because it does not require passing information inside the swarm about the objective values.
        # In case such information is stored, the current implementation may be changed to that.
        # The paper itself, however, does NOT make it clear which one of the two it is referring to.

        d_g, d_min, d_max = d[g], np.min(d), np.max(d)

        # NOTE: The evolutionary factor is bound in the values [0,1].
        evolutionary_factor = (d_max - d_g) / (d_max - d_min)
        return evolutionary_factor




    
    def __average_between_each_particle_to_all_others(self):
        """
        Returns
        -------
        : np.array
            shape = (n, ), where n is the number of rows of the "self.swarm_positions" attribute
            (which in turn encodes the number of particles of the swarm).

            output[i] := average distance (2-norm) of particle i w.r.t. all other particles of the swarm.
        """
        # particle_difference_vectors[particle i][particle j][dimension d] := difference vector between particles i and j (i-j) 
        particle_difference_vectors = np.apply_along_axis(lambda row: row - self.swarm_positions, arr=self.swarm_positions, axis=1)
        two_norm_of_vector_i_j = np.linalg.norm(particle_difference_vectors, ord=2, axis=-1)
        # Math: d_i = \frac{1}{N-1} \sum_{i=1}^{N} \| \mathbf{x}_i - \mathbf{x}_j \|
        average_distances_between_i_j = np.sum(two_norm_of_vector_i_j, axis=1) / two_norm_of_vector_i_j.shape[0]

        return average_distances_between_i_j




    

class AdaptivePSO():

    def __init__(self, is_adaptive: bool = False):
        if isinstance(is_adaptive, bool): # Sanity check
            self._is_adaptive = is_adaptive
            self._evolutionary_state = None
        else:
            raise TypeError(f"The input parameter is_adaptive must be of type {type(bool)}.\n\
            {type(is_adaptive)} was provided instead.")
        
    def _classify_evolutionary_state(self, f_evol):
        # Sanity check
        if (isinstance(f_evol, numbers.Real)):
            return classify_evolutionary_state(f_evol)
        else:
            raise TypeError(f"Expected type of parameter evolutionary factor f_evol to be rational number {type(numbers.Real)}.\n\
            {type(f_evol)} was provided instead.")




    def _apply_eliticism_learning_strategy(self, evolutionary_state: EvolutionaryStates):
        # This strategy aims at maintaining the diversity during the search,
        # by causing a escape state of the best particle when it gets trapped in a local minimum.
        # That is why this strategy is executed only in case of convergence.
        if evolutionary_state == EvolutionaryStates.CONVERGENCE:
            # Picking a dimension at random for the mutation to take place.
            search_space_dimension = randrange(len(self._spawn_boundaries))

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
