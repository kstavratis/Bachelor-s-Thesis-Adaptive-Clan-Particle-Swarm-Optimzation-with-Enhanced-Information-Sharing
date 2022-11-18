from numpy.linalg import norm
from numpy import e
from enum import Enum, auto
import numbers
from random import uniform, gauss, randrange

from .evolutionary_state_classification_singleton_method import classify_evolutionary_state
from .enums.evolutionary_states import EvolutionaryStates

from classes.PSOs.particle import Particle

c_min, c_max = 1.5, 2.5 # We'll have to see whether this line of code is executed according to the import command. e.g. import only class, is this being executed?


class AdaptivePSO(object):

    def __init__(self, is_adaptive: bool = False):
        if isinstance(is_adaptive, bool): # Sanity check
            self._is_adaptive = is_adaptive
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


    def _estimate_evolutionary_state(self):
        if not self._is_adaptive:
            raise ValueError("This PSO instantiation is not adaptive.")
        return self.__evaluate_evolutionary_factor(self.__evaluate_average_between_each_particle_to_all_others())

        ...

    def __evaluate_average_between_each_particle_to_all_others(self):
        d = []
        # # Multi-threaded version
            # processes = []
            # for particle in self.swarm:
            #     p = Process(target=self.__calculate_average_distance_of_particle_from_all_others, args=(d,particle))
            #     p.start()
            #     processes.append(p)
            #
            # for process in processes:
            #     process.join()


        # Single-threaded version.
        for particle1 in self.swarm:
            # "Adaptive Clan Particle Swarm Optimization" -> equation (3), i.e.
            # d_i = 1/(N-1) * Σ||x_i - x_j||
            d.append(
                1 / (len(self.swarm) - 1) * sum(norm(particle1._position - particle2._position)
                                                for particle2 in self.swarm)
            )

        return d

        # particle_positions = array([particle._position for particle in self.swarm])
        # @njit(parallel=True)
        # def numba_implementation(particle_positions: array):
        #     swarm_size = len(particle_positions)
        #     d = zeros(swarm_size)
        #     for particle1 in prange(swarm_size):
        #         for particle2 in range(swarm_size):
        #             d[particle1] += norm(
        #                 particle_positions[particle1] - particle_positions[particle2]
        #             )
        #         print(d)
        #     d /= swarm_size - 1
        #     print(d)
        #     return d
        #
        # return numba_implementation(particle_positions)

    
    def __evaluate_evolutionary_factor(self, d: list) -> float:
        # TODO: Is use of the command: "best_particle_of_the_swarm = self.__particle_with_best_personal_best"
        #       WRONG in this case?
        #       'particle_with_best_personal_best' returns the particle which AT SOME POINT had the best phenotype
        #       found during the (up until that point) execution of the algorithm, not the particle which
        #       CURRENTLY has the best phenotype. These two can differ, because the CURRENT best value
        #       may be worse than the best value to has ever been found, but the latter was 'lost'
        #       due to the movement of the particles.
        #       That would leave 3 points of interest:
        #       1) gbest
        #       2) particle which *at some point* had gbest, but doesn't at the current iteration
        #       3) particle which *at the current iteration* has the best phenotype, which is worse than f(gbest).
        #       The point of the procedure is to estimate distribution of PARTICLES, therefore following
        #       option 1) might be a false approach.
        #       In the end, I select option 2), but it might be a point of contingency.

        # best_particle_of_the_swarm = min(self.swarm) # Option 3). Requires a comparison operator among the set {lt, le, gt, ge} to have been overloaded.
        best_particle_of_the_swarm = self._particle_with_best_personal_best

        d_g = 1 / (len(self.swarm) - 1) * \
                sum(norm(best_particle_of_the_swarm._position - particle._position)
                    for particle in self.swarm)

        #! The only case this is possible if "best_particle_of_the_swarm" does not match any particle of the "self.swarm",
        #! due to some bug.
        if d_g not in d:
            raise ValueError("Average distance d_g does not match any of the average distances d previously calculated.")

        d_min, d_max = min(d), max(d)

        # Note that this value (the evolutionary factor) is bounded in the values [0,1].
        return (d_g - d_min) / (d_max - d_min)


    
    def _determine_accelaration_coefficients(self, evolutionary_state: EvolutionaryStates):

        acceleration_rate = uniform(0.05, 0.10)
        c1_strategy, c2_strategy = "", ""
        if evolutionary_state == EvolutionaryStates.EXPLORATION:
            c1_strategy = _CoefficientOperations.INCREASE
            c2_strategy = _CoefficientOperations.DECREASE
        elif evolutionary_state == EvolutionaryStates.EXPLOITATION:
            c1_strategy = _CoefficientOperations.INCREASE_SLIGHTLY
            c2_strategy = _CoefficientOperations.DECREASE_SLIGHTLY
        elif evolutionary_state == EvolutionaryStates.CONVERGENCE:
            c1_strategy = _CoefficientOperations.INCREASE_SLIGHTLY
            c2_strategy = _CoefficientOperations.INCREASE_SLIGHTLY
        elif evolutionary_state == EvolutionaryStates.JUMP_OUT:
            c1_strategy = _CoefficientOperations.DECREASE
            c2_strategy = _CoefficientOperations.INCREASE
        else:
            raise ValueError(f"The evolutionary state can only be in one of the {len(EvolutionaryStates)} allowed states {[state for state in EvolutionaryStates.__members__.keys()]}.\n\
            The 'evolutionary_state' parameter has the value {evolutionary_state} instead.")

        if c1_strategy == _CoefficientOperations.INCREASE:
            if self.c1 + acceleration_rate <= c_max:
                self.c1 += acceleration_rate
        elif c1_strategy == _CoefficientOperations.INCREASE_SLIGHTLY:
            if self.c1 + 0.5 * acceleration_rate <= c_max:
                self.c1 += 0.5 * acceleration_rate
        elif c1_strategy == _CoefficientOperations.DECREASE:
            if self.c1 - acceleration_rate >= c_min:
                self.c1 -= acceleration_rate

        if c2_strategy == _CoefficientOperations.INCREASE:
            if self.c2 + acceleration_rate <= c_max:
                self.c2 += acceleration_rate
        elif c2_strategy == _CoefficientOperations.INCREASE_SLIGHTLY:
            if self.c2 + 0.5 * acceleration_rate <= c_max:
                self.c2 += 0.5 * acceleration_rate
        elif c2_strategy == _CoefficientOperations.DECREASE:
            if self.c2 - acceleration_rate >= c_min:
                self.c2 -= acceleration_rate
        elif c2_strategy == _CoefficientOperations.DECREASE_SLIGHTLY:
            if self.c2 - 0.5 * acceleration_rate >= c_min:
                self.c2 -= 0.5 * acceleration_rate

        # In order to avoid an explosion state, it is necessary to bound the sum c1+c2.
        # As the minimum and the maximum value for c1 and c2 are c_min = 1.5 and c_max = 2.5,
        # when c1 + c2 > cm + cmax, one should update c1 and c2 using:
        # c_i = c_i/(c1 + c2) (c_min + c_max):
        if sum([self.c1, self.c2]) > c_min + c_max:
            c1_old, c2_old = self.c1, self.c2
            self.c1 = c1_old / (c1_old + c2_old) * (c_min + c_max)
            self.c2 = c2_old / (c1_old + c2_old) * (c_min + c_max)



    def _apply_eliticism_learning_strategy(self, evolutionary_state: EvolutionaryStates):
        # This strategy aims at maintaining the diversity during the search,
        # by causing a escape state of the best particle when it gets trapped in a local minimum.
        # That is why this strategy is executed only in case of convergence.
        if evolutionary_state == EvolutionaryStates.CONVERGENCE:
            # Picking a dimension at random for the mutation to take place.
            search_space_dimension = randrange(len(self.__spawn_boundaries))

            search_space_range = \
                self.__spawn_boundaries[search_space_dimension][1] - self.__spawn_boundaries[search_space_dimension][0]
            # "Adaptive Particle Swarm Optimization, Zhi et al." -> "IV. APSO" -> "C. ELS" ->
            # "Empirical study shows that σ_max = 1.0 and σ_min = 0.1
            # result in good performance on most of the test functions"
            sigma = (1 - (1 - 0.1) / self.__max_iterations * self.current_iteration)
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
                    self.__spawn_boundaries[search_space_dimension][0]:
                mutated_position[search_space_dimension] = self.__spawn_boundaries[search_space_dimension][0]
            elif self._particle_with_best_personal_best._position[search_space_dimension] + search_space_range * elitist_learning_rate > \
                self.__spawn_boundaries[search_space_dimension][1]:
                mutated_position[search_space_dimension] = self.__spawn_boundaries[search_space_dimension][1]
            else:
                mutated_position[search_space_dimension] += search_space_range * elitist_learning_rate

            # If the mutated position achieves a better fitness function, then have the best particle move there.
            # Note: This PSO implementation follows a minimization approach. Therefore, "better" is equivalent to "lower value".
            if Particle.fitness_function(mutated_position) < Particle.fitness_function(self._particle_with_best_personal_best._position):
                self._particle_with_best_personal_best._position = mutated_position
            else:  # Replacing particle with worst position with particle with mutated best position.
                current_worst_particle = self.__find_particle_with_best_personal_best(greater_than=True)
                self.swarm.remove(current_worst_particle)
                self.swarm.append(Particle(Particle.fitness_function, self.__spawn_boundaries, spawn_position=mutated_position))

    def _adapt_inertia_factor(self, f_evol: float):
        self.w = 1 / (1 + 1.5 * e ** (-2.6 * f_evol))  # ∈[0.4, 0.9]  ∀f_evol ∈[0,1]


class _CoefficientOperations(Enum):
    INCREASE = auto(),
    DECREASE = auto(),
    INCREASE_SLIGHTLY = auto(),
    DECREASE_SLIGHTLY = auto()