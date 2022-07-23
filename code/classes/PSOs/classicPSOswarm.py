"""
Copyright (C) 2022  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from numpy.ma import sqrt
from random import uniform, randrange, gauss
from numpy import mean, e, diag
from numpy.linalg import norm
from types import FunctionType
from enum import Enum, auto
from typing import List, Tuple#, Final


from classes.PSOs.particle import Particle
from classes.enums.evolutionary_states import EvolutionaryStates
from classes.enums.wall_types import WallTypes
from classes.enums.enhanced_information_sharing.global_local_coefficient_types import GlobalLocalCoefficientTypes
from classes.enums.enhanced_information_sharing.control_factor_types import ControlFactorTypes
from scripts.evolutionary_state_classification_singleton_method import classify_evolutionary_state


# To achieve a balance between global and local exploration to speed up convergence to the true optimum,
# an inertia weight whose value decreases linearly with the iteration number has been used.
# The values of w_min = 0.4 and w_max = 0.9 are widely used.
w_min, w_max = 0.4, 0.9
c_min, c_max = 1.5, 2.5
c3_start, c3_end = 2, 0

# PARTICLE_SIMILARITY_LIMIT : Final = 10**(-9)
PARTICLE_SIMILARITY_LIMIT = 10 ** (-20)


# TODO: Python code optimization comment.
#   Admittedly, this file is following a policy which encumbers Python for the sake of code readability.
#   In the functions below, a lot of nested functions are defined, which makes Python create function objects
#   at each iteration of the algorithm. These could be rewritten with one global definition as private functions.
#   Currently, readability is considered to be a major benefit for following said policy.


class ClassicSwarm:
    def __init__(self, swarm_or_fitness_function, spawn_boundaries: list,
                 maximum_iterations: int,
                 # w: float,
                 swarm_size: int = 50,
                 c1: float = 2, c2: float = 2,
                 adaptivePSO: bool = False,
                 eis: Tuple[
                     Tuple[GlobalLocalCoefficientTypes, float or None], Tuple[ControlFactorTypes, float or None]] =
                 ((GlobalLocalCoefficientTypes.NONE, None), (ControlFactorTypes.NONE, None)),
                 current_iteration: int = 0,
                 search_and_velocity_boundaries: List[List[float]] = None, wt: WallTypes = WallTypes.NONE):
        if isinstance(swarm_or_fitness_function, FunctionType):
            self.swarm = [Particle(swarm_or_fitness_function, spawn_boundaries) for _ in range(swarm_size)]
        if isinstance(swarm_or_fitness_function, list):
            self.swarm = swarm_or_fitness_function
        if adaptivePSO or search_and_velocity_boundaries is not None:
            # The convex boundaries need to be stored in order to apply
            # the learning strategy (using eliticism)
            # For details, see -> "Adaptive Clan Particle Swarm Optimization" ->
            # -> "III. ADAPTIVE PARTICLE SWARM OPTIMIZATION" -> "D. Learning Strategy using Elitism"
            self.__spawn_boundaries = spawn_boundaries
        if search_and_velocity_boundaries is not None:
            self.__wall_type = wt

        # Initializing the fitness_function's global best _position particle with the optimal value on spawn.
        self.__fitness_function = swarm_or_fitness_function

        # The best particle is stored as a local variable (pointer), as it is required in many stages of the algorithm
        # This is expected to enhance runtime performance by cutting down on objective function evaluations.
        self.__particle_with_best_personal_best = self.__find_particle_with_best_personal_best()
        self.global_best_position = self._find_global_best_position()  # Could become a dynamically-added field.
        # Storing the learning rates c1, c2.
        # Both are shared among all particles of the swarm.
        self.c1, self.c2 = c1, c2

        # Handling existence of Enhanced Information Sharing (EIS).
        global_locaL_coefficient_method, c3, control_factor_method, c3_k = eis[0][0], eis[0][1], eis[1][0], eis[1][1]

        self._control_factor_method = control_factor_method
        self._global_local_coefficient_method = global_locaL_coefficient_method

        # c3 is not a number <=> c3_k is not a number
        if self._global_local_coefficient_method == GlobalLocalCoefficientTypes.NONE:
            self.c3, self.c3_k = None, None


        elif self._global_local_coefficient_method == GlobalLocalCoefficientTypes.CONSTANT:
            if isinstance(c3, float):
                self.c3 = c3
            else:
                raise ValueError("Global-local coefficient c3 that was given is not a real number.")

        elif self._global_local_coefficient_method == GlobalLocalCoefficientTypes.LINEAR:
            self.c3 = c3  # Note that in reality, c3 is calculated dynamically at each iteration.

        elif self._global_local_coefficient_method == GlobalLocalCoefficientTypes.ADAPTIVE:
            if isinstance(c3, float):
                self.c3 = c3
            else:
                raise ValueError("Global-local coefficient c3 that was given is not a real number.")
            self.c3_k = None  # c3 is dependent on c3_k only in the case of a linear c3.

        else:
            ValueError("A not defined Enhanced Information Sharing (EIS) methodology has been given.")

        # c3 is a number <=> c3_k is a number
        if self._global_local_coefficient_method == GlobalLocalCoefficientTypes.CONSTANT or \
                self._global_local_coefficient_method == GlobalLocalCoefficientTypes.LINEAR or \
                self._global_local_coefficient_method == GlobalLocalCoefficientTypes.ADAPTIVE:
            if isinstance(c3_k, float):
                self.c3_k = c3_k
            else:
                raise ValueError("The value assigned to the control factor c3_k is not a floating-point number.")

        if self._control_factor_method == ControlFactorTypes.LINEAR:
            self.__c3_k_start = c3_k


        # Note that in all PSO variations used, inertia weight "w" is calculated dynamically
        # in the "update_parameters" function.

        self.__adaptivePSO = adaptivePSO
        self.__max_iterations, self.current_iteration = maximum_iterations, current_iteration
        self.__domain_and_velocity_boundaries = search_and_velocity_boundaries
        # Note: It is good practice to have the speed limits be a multiple of the domain limits.
        # In the article "Adaptive Particle Swarm Optimization", Zhan et al. (https://ieeexplore.ieee.org/document/4812104)
        # used a Vmax of 20% the domain limits in each dimension.
        # Adaptive Particle Swarm Optimization -> II. PSO AND ITS DEVELOPMENTS -> A. PSO Framework

    def __find_particle_with_best_personal_best(self, greater_than: bool = False) -> Particle:
        """
        Particle that at some point had the best-known position (global min).
        Take into account the possibility that no particle is on that position when this function is called.
        This can happen when the particle which found this position moved to a worse position.

        :param greater_than: Returns particle with greatest position if True . Returns particle with lowest position if False (Default).

        :return:
        """

        # TODO: While the implementation below is computationally correct, it has one main weakness:
        #   it recalculates the fitness value for each point, which may be computationally expensive,
        #   such as in the case of the Quadric function, where the cost of computing a value is O(n^2).
        #   This principle may be generalized into other computationally expensive problems (fitness function values),
        #   such as calculating the error of a neural network training cycle.
        #   POSSIBLE SOLUTION: Store the best particle population at runtime, using either a pointer to the
        #   particle object or an index to its position in the array/list.


        # Initializing it into the 1st particle for comparison.
        particle_with_best_personal_best = self.swarm[0]
        for particle in self.swarm:
            # This implementation follows the standard convention that PSO is tasked to solve a minimization problem.
            # Therefore, the default (if not only) behaviour is to consider the lower fitness value as the better solution.

            # Maximization problem.
            if greater_than:
                if Particle.fitness_function(particle._personal_best_position) \
                        > \
                        Particle.fitness_function(particle_with_best_personal_best._personal_best_position):
                    particle_with_best_personal_best = particle
            # Minimization problem (Default).
            else:
                if Particle.fitness_function(particle._personal_best_position) \
                        < \
                        Particle.fitness_function(particle_with_best_personal_best._personal_best_position):
                    particle_with_best_personal_best = particle

        return particle_with_best_personal_best

    def _find_global_best_position(self):
        """
        Best position that the swarm has found during execution of the algorithm.
        :return:
        """
        return self.__particle_with_best_personal_best._personal_best_position

    def update_swarm(self):

        def wall_enforcement():

            def replace():
                for particle in self.swarm:
                    for axis in particle._position:
                        if not (self.__domain_and_velocity_boundaries[0][0] <
                                axis < self.__domain_and_velocity_boundaries[0][1]):
                            # Remove this particle, which has exceeded the allowed boundaries.
                            self.swarm.remove(particle)
                            # Replace the deleted particle with a new one.
                            self.swarm.append(Particle(Particle.fitness_function, self.__spawn_boundaries))
                            break

            def absorb():
                for particle in self.swarm:
                    for axis in range(len(particle._position)):  # == len(particle.__velocity)
                        if not (self.__domain_and_velocity_boundaries[0][0] <
                                particle._position[axis] < self.__domain_and_velocity_boundaries[0][1]):
                            particle._zero_velocity(axis)

            def reflect():
                for particle in self.swarm:
                    for axis in range(len(particle._position)):  # == len(particle.__velocity)
                        if not (self.__domain_and_velocity_boundaries[0][0] <
                                axis < self.__domain_and_velocity_boundaries[0][1]):
                            particle._reflect_velocity(axis)

            # TODO develop invisible wall (don't calculate fitness value of particles outside the allowed domain).
            def ignore():
                return

            if self.__wall_type == WallTypes.NONE:
                return
            elif self.__wall_type == WallTypes.ELIMINATING:
                replace()
            elif self.__wall_type == WallTypes.ABSORBING:
                absorb()
            elif self.__wall_type == WallTypes.REFLECTING:
                reflect()
            elif self.__wall_type == WallTypes.INVISIBLE:
                ignore()
            else:
                raise ValueError("Undefined wall type has been given.")

        def limit_particle_velocity():
            for particle in self.swarm:
                particle._limit_velocity(self.__domain_and_velocity_boundaries[1])

        def replace_similar_particles(gamma: float = 0.01):
            """
            Algorithm
            ---------
            Step 1: Calculate average distances between each particle to all others.
            Step 2: Calculate mean (μ) and standard deviation (σ) of Step 1.
            Step 3: Remove all particles in between [ μ - γ σ, μ + γ σ ], γ > 0  (68-95-99.7 rule) and replace them with new ones.
            :param gamma: constant determining the range from the mean μ mentioned in Step 2.
            """

            # # Step 1
            # average_distances = []
            # for particle_i in range(len(self.swarm)):
            #     average_distance_of_particle_i = sum(norm(self.swarm[particle_i]._position - particle_j._position)
            #                                          for particle_j in self.swarm) / len(self.swarm)
            #     average_distances.append(average_distance_of_particle_i)
            #
            # # Step 2
            # mu, sigma = mean(average_distances), std(average_distances)
            #
            # # Step 3
            # for particle_index in range(len(self.swarm)):
            #     if mu - gamma * sigma < average_distances[particle_index] < mu + gamma * sigma:
            #         print("Deleting particle at position " + str(particle_index))
            #         del self.swarm[particle_index]
            #         print("Inserting new particle at position " + str(particle_index))
            #         self.swarm.insert(particle_index, Particle(self.__fitness_function, self.__spawn_boundaries))

            for particle in self.swarm:
                for other_particle in self.swarm:
                    if other_particle != particle:  # Obviously the distance/similarity between a particle and itself is 0.
                        similarity = norm(particle._position - other_particle._position)
                        if similarity < PARTICLE_SIMILARITY_LIMIT:
                            # Delete this particle, since it is (very) similar to a different particle.
                            self.swarm.remove(particle)
                            # Replace the deleted particle with a new one.
                            self.swarm.append(Particle(self.__fitness_function, self.__spawn_boundaries))
                            break

        self.__update_parameters()

        random_multipliers = tuple(
            [
                diag([uniform(0,1) for _ in range(len(self.__spawn_boundaries))]),  # r1
                diag([uniform(0,1) for _ in range(len(self.__spawn_boundaries))]),  # r2
                None                                                                # r3 (potentially)
            ]
            for i in range(len(self.swarm))
        )

        if isinstance(self.c3, float):
            for triad in random_multipliers:
                triad[2] = diag([uniform(0,1) for _ in range(len(self.__spawn_boundaries))])

        random_multipliers_index = 0
        for particle in self.swarm:
            particle._update_position(self.global_best_position,
                                      self.w,
                                      self.c1, random_multipliers[random_multipliers_index][0],
                                      self.c2, random_multipliers[random_multipliers_index][1],
                                      self.c3, random_multipliers[random_multipliers_index][2], self._control_factor_method)
            random_multipliers_index += 1

        # This is executed only when the Last-Elimination Principle is enabled. TODO: Why only then?
        if self.__domain_and_velocity_boundaries is not None:
            wall_enforcement()
            limit_particle_velocity()
            # replace_similar_particles()

        # Calculating the new best particle and position of the swarm.
        self.__particle_with_best_personal_best = self.__find_particle_with_best_personal_best()
        self.global_best_position = self._find_global_best_position()


        self.current_iteration += 1

    def __update_parameters(self):

        def update_pso_inertia_weight_and_coefficients():
            if self.__adaptivePSO:
                f_evol = self.__estimate_evolutionary_state()
                evolutionary_state = classify_evolutionary_state(f_evol)

                self.__determine_accelaration_coefficients(evolutionary_state)
                # self.__apply_eliticism_learning_strategy(evolutionary_state)
                self.__adapt_inertia_factor(f_evol)
            else:  # Follow classic PSO learning strategy: decrease inertia weight linearly.
                self.w = w_max - ((w_max - w_min) / self.__max_iterations) * self.current_iteration

        def update_global_local():

            def update_global_local_coefficient():
                # Changing global-local coefficient c_3
                if self._global_local_coefficient_method == GlobalLocalCoefficientTypes.NONE or \
                        self._global_local_coefficient_method == GlobalLocalCoefficientTypes.CONSTANT:
                    pass
                # Global optimization capability is strong when c_3 is linearly decreasing (c3_k > 0) according to the article.
                elif self._global_local_coefficient_method == GlobalLocalCoefficientTypes.LINEAR:
                    self.c3 = self.c3_k * (
                                c3_start - (c3_start - c3_end) / self.__max_iterations * self.current_iteration)
                elif self._global_local_coefficient_method == GlobalLocalCoefficientTypes.ADAPTIVE:
                    pass  # It is handled in the "determine_accelaration_coefficients" function called above (see "if self adaptivePSO").

            def update_global_local_control_factor():
                if self._control_factor_method == ControlFactorTypes.NONE or \
                        self._control_factor_method == ControlFactorTypes.CONSTANT:
                    pass

                # Changing global-local coefficient control factor k.
                elif self._control_factor_method == ControlFactorTypes.LINEAR:
                    self.c3_k = self.__c3_k_start - (
                                self.__c3_k_start - 0) / self.__max_iterations * self.current_iteration

                elif self._control_factor_method == ControlFactorTypes.ADAPTIVE:
                    pass  # handled in "__determince accelaration coefficients" function.

            update_global_local_coefficient()
            update_global_local_control_factor()

        # Executing the two functions defined above.
        update_pso_inertia_weight_and_coefficients()
        update_global_local()

    def calculate_swarm_distance_from_swarm_centroid(self):
        swarm_positions = [self.swarm[i]._position for i in range(len(self.swarm))]
        swarm_centroid = mean(swarm_positions, axis=0)
        # SD = √(1/N * Σ(||x_avg - x_i||^2))
        swarm_standard_deviation = sqrt(sum(norm(
            swarm_centroid - swarm_positions[i]) ** 2 for i in range(len(swarm_positions))) / len(self.swarm))
        return swarm_standard_deviation

    def __estimate_evolutionary_state(self):
        """
        Function to be used in APSO (Adaptive Particle Swarm Optimization).
        Returns the evolutionary factor "f_evol".

        "In order to adjust properly the acceleration parameters,
        one should estimate the evolutionary state of the swarm.
        This process is divided in two separate steps.
        In the first, one needs to evaluate the average distance between each particle to all the others.
        After this, one should evaluate the the evolutionary factor (f_evol)."
        :return: "f_evol": float ∈[0,1]
        """



        def evaluate_average_between_each_particle_to_all_others():
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
                # "Adaptive Clan Particle Swarm Optimization" -> equation (3)
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

        def evaluate_evolutionary_factor(d: list) -> float:
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
            best_particle_of_the_swarm = self.__particle_with_best_personal_best

            d_g = 1 / (len(self.swarm) - 1) * \
                  sum(norm(best_particle_of_the_swarm._position - particle._position)
                      for particle in self.swarm)

            if d_g not in d:
                raise ValueError("Average distance d_g does not match any of the average distances d previously calculated.")

            d_min, d_max = min(d), max(d)

            # Note that this value (the evolutionary factor) is bounded in the values [0,1].
            return (d_g - d_min) / (d_max - d_min)

        return evaluate_evolutionary_factor(evaluate_average_between_each_particle_to_all_others())

    # # Needed for multi-threaded calculation of average distances d_i.
    # def __calculate_average_distance_of_particle_from_all_others(self, d: list, particle: Particle):
    #     d.append(
    #         1 / (len(self.swarm) - 1) * sum(norm(particle._position - other_particle._position)
    #                                         for other_particle in self.swarm)
    #     )

    def __determine_accelaration_coefficients(self, evolutionary_state: EvolutionaryStates):

        class CoefficientOperations(Enum):
            INCREASE = auto(),
            DECREASE = auto(),
            INCREASE_SLIGHTLY = auto(),
            DECREASE_SLIGHTLY = auto()

        acceleration_rate = uniform(0.05, 0.10)
        c1_strategy, c2_strategy = "", ""
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
            raise ValueError("The evolutionary state can only be in one of the (four) allowed states.")

        if c1_strategy == CoefficientOperations.INCREASE:
            if self.c1 + acceleration_rate <= c_max:
                self.c1 += acceleration_rate
        elif c1_strategy == CoefficientOperations.INCREASE_SLIGHTLY:
            if self.c1 + 0.5 * acceleration_rate <= c_max:
                self.c1 += 0.5 * acceleration_rate
        elif c1_strategy == CoefficientOperations.DECREASE:
            if self.c1 - acceleration_rate >= c_min:
                self.c1 -= acceleration_rate

        if c2_strategy == CoefficientOperations.INCREASE:
            if self.c2 + acceleration_rate <= c_max:
                self.c2 += acceleration_rate
        elif c2_strategy == CoefficientOperations.INCREASE_SLIGHTLY:
            if self.c2 + 0.5 * acceleration_rate <= c_max:
                self.c2 += 0.5 * acceleration_rate
        elif c2_strategy == CoefficientOperations.DECREASE:
            if self.c2 - acceleration_rate >= c_min:
                self.c2 -= acceleration_rate
        elif c2_strategy == CoefficientOperations.DECREASE_SLIGHTLY:
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

        if self._global_local_coefficient_method == GlobalLocalCoefficientTypes.ADAPTIVE:
            # When local exploration is encouraged, the coefficient contributes a vector starting
            # from the swarm's global best pointing towards the particle's local best.
            if evolutionary_state == EvolutionaryStates.EXPLORATION or \
                    evolutionary_state == EvolutionaryStates.EXPLOITATION:
                self.c3 = -(abs(self.c3))
            # When global exploration is encouraged, the coefficient contributes a vector starting
            # from the particle's best pointing towards the swarm's global best.
            elif evolutionary_state == EvolutionaryStates.CONVERGENCE or \
                    evolutionary_state == EvolutionaryStates.JUMP_OUT:
                self.c3 = abs(self.c3)

        if self._control_factor_method == ControlFactorTypes.ADAPTIVE:
            # When local exploration is encouraged, the coefficient contributes a vector starting
            # from the swarm's global best pointing towards the particle's local best.
            if evolutionary_state == EvolutionaryStates.EXPLORATION or \
                    evolutionary_state == EvolutionaryStates.EXPLOITATION:
                self.c3_k = -(abs(self.c3_k))
            # When global exploration is encouraged, the coefficient contributes a vector starting
            # from the particle's best pointing towards the swarm's global best.
            elif evolutionary_state == EvolutionaryStates.CONVERGENCE or \
                    evolutionary_state == EvolutionaryStates.JUMP_OUT:
                self.c3_k = abs(self.c3_k)



    def __apply_eliticism_learning_strategy(self, evolutionary_state: EvolutionaryStates):
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
            mutated_position = self.__particle_with_best_personal_best._position

            # Making sure that the particle will still be inside the search space after the mutation.
            # First two conditions (if and elif) enforce that the particle remains inside the search boundaries where they to be exceeded.
            # Last one (else) applies the calculated transposition as is.
            # TODO Create a better strategy for ensuring that the particle remains in search boundaries.
            #  Attempt to generate a new random number (always following the same distribution) until a valid one
            #  (one that keeps the particle inside the search space) is generated was made, but experimentally,
            #  it showed that this can slow down the PSO significantly, because of its random nature, especially in the
            #  early stages of the algorithm. Cutting down mutation step range to "search_space_range/2" attempt was made
            #  but it did not lead to a satisfying speed-up.
            if self.__particle_with_best_personal_best._position[search_space_dimension] + search_space_range * elitist_learning_rate < \
                    self.__spawn_boundaries[search_space_dimension][0]:
                mutated_position[search_space_dimension] = self.__spawn_boundaries[search_space_dimension][0]
            elif self.__particle_with_best_personal_best._position[search_space_dimension] + search_space_range * elitist_learning_rate > \
                self.__spawn_boundaries[search_space_dimension][1]:
                mutated_position[search_space_dimension] = self.__spawn_boundaries[search_space_dimension][1]
            else:
                mutated_position[search_space_dimension] += search_space_range * elitist_learning_rate

            # If the mutated position achieves a better fitness function, then have the best particle move there.
            # Note: This PSO implementation follows a minimization approach. Therefore, "better" is equivalent to "lower value".
            if Particle.fitness_function(mutated_position) < Particle.fitness_function(current_best_particle._position):
                current_best_particle._position = mutated_position
            else:  # Replacing particle with worst position with particle with mutated best position.
                current_worst_particle = self.__find_particle_with_best_personal_best(greater_than=True)
                self.swarm.remove(current_worst_particle)
                self.swarm.append(Particle(Particle.fitness_function, self.__spawn_boundaries, spawn_position=mutated_position))

    def __adapt_inertia_factor(self, f_evol: float):
        self.w = 1 / (1 + 1.5 * e ** (-2.6 * f_evol))  # ∈[0.4, 0.9]  ∀f_evol ∈[0,1]
