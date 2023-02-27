"""
Copyright (C) 2022  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from numpy.ma import sqrt
from random import uniform
from numpy import mean, e, diag
from numpy.linalg import norm
from types import FunctionType
from enum import Enum, auto
from typing import List, Tuple#, Final


# from classes.PSOs.particle import Particle
from .particle import Particle
# from classes.PSOs.Mixins.adaptive.adaptive import AdaptivePSO
from .Mixins.adaptive.adaptive import AdaptivePSO
from .Mixins.enhanced_information_sharing.enhanced_information_sharing import EnhancedInformationSharingPSO
from .Mixins.adaptive.enums.evolutionary_states import EvolutionaryStates
from .wall_types import WallTypes
from .Mixins.enhanced_information_sharing.enums.global_local_coefficient_types import GlobalLocalCoefficientTypes
from .Mixins.enhanced_information_sharing.enums.control_factor_types import ControlFactorTypes
from .Mixins.adaptive.evolutionary_state_classification_singleton_method import classify_evolutionary_state


# To achieve a balance between global and local exploration to speed up convergence to the true optimum,
# an inertia weight whose value decreases linearly with the iteration number has been used.
# The values of w_min = 0.4 and w_max = 0.9 are widely used.
w_min, w_max = 0.4, 0.9
# c_min, c_max = 1.5, 2.5
# c3_start, c3_end = 2.0, 0.0

# PARTICLE_SIMILARITY_LIMIT : Final = 10**(-9)
PARTICLE_SIMILARITY_LIMIT = 10 ** (-20)


# TODO: Python code optimization comment.
#   Admittedly, this file is following a policy which encumbers Python for the sake of code readability.
#   In the functions below, a lot of nested functions are defined, which makes Python create function objects
#   at each iteration of the algorithm. These could be rewritten with one global definition as private functions.
#   Currently, readability is considered to be a major benefit for following said policy.


class ClassicSwarm(AdaptivePSO, EnhancedInformationSharingPSO):
    def __init__(self,
                swarm_or_fitness_function,
                spawn_boundaries: list,
                maximum_iterations: int,
                # w: float,
                swarm_size: int = 50,
                c1: float = 2, c2: float = 2,
                is_adaptive: bool = False,
                eis: Tuple[
                     Tuple[GlobalLocalCoefficientTypes, float or None], Tuple[ControlFactorTypes, float or None]] =
                 ((GlobalLocalCoefficientTypes.NONE, None), (ControlFactorTypes.NONE, None)),
                current_iteration: int = 0,
                search_and_velocity_boundaries: List[List[float]] = None, wt: WallTypes = WallTypes.NONE):
        if isinstance(swarm_or_fitness_function, FunctionType):
            self.swarm = [Particle(swarm_or_fitness_function, spawn_boundaries) for _ in range(swarm_size)]
        if isinstance(swarm_or_fitness_function, list):
            self.swarm = swarm_or_fitness_function


        # Initializing the fitness_function's global best _position particle with the optimal value on spawn.
        self.__fitness_function = swarm_or_fitness_function

        # The best particle is stored as a local variable (pointer), as it is required in many stages of the algorithm.
        # This is expected to enhance runtime performance by cutting down on objective function evaluations.

        self._particle_with_best_personal_best = self._find_particle_with_best_personal_best()
        self.global_best_position = self._particle_with_best_personal_best._personal_best_position
        self.f_global_best = Particle.fitness_function(self.global_best_position)
        # Storing the learning rates c1, c2.
        # Both are shared among all particles of the swarm.
        self.c1, self.c2 = c1, c2

        

        #* Note that in all PSO variations used, inertia weight "w" is calculated dynamically
        #* in the "update_parameters" function.

        #! Software engineering note: It would be a better practice to define "current_iteration"
        #! as a private variable and implement (trivial) getter and incrementer public functions.
        #! This way, object encapsulation is achieved.
        #! To demonstrate, malicious client may manually frequently change (decrease) the value of "current_iteration"
        #! in such a way that the termination condition "self._max_iterations == self.current_iteration" of the main
        #! algorithm may never be satisfied.
        #! In the case which:
        #!  1) the distribution of the particles of the swarm is "sparse" enough such that Underflow Error doesn't occur.
        #!  2) the swarm does not satisfyingly converge to the global optimum, because it has been trapped in a local optimum.
        #! the program may encounter an infinite loop!
        self._max_iterations, self.current_iteration = maximum_iterations, current_iteration
        self.__domain_and_velocity_boundaries = search_and_velocity_boundaries
        # Note: It is good practice to have the speed limits be a multiple of the domain limits.
        # In the article "Adaptive Particle Swarm Optimization", Zhan et al. (https://ieeexplore.ieee.org/document/4812104)
        # used a Vmax of 20% the domain limits in each dimension.
        # Adaptive Particle Swarm Optimization -> II. PSO AND ITS DEVELOPMENTS -> A. PSO Framework


        # TODO : Look into the "cooperative multiple inheritance" pattern to potentially call __init__ of all parent classes/mixins.
        # super(ClassicSwarm, AdaptivePSO).__init__(is_adaptive) # AdaptivePSO
        AdaptivePSO.__init__(self, is_adaptive) 
        # super(ClassicSwarm, EnhancedInformationSharingPSO).__init__(True, eis[0][0], eis[0][1], eis[1][0], eis[1][1]) # Enhanced Information Sharing
        EnhancedInformationSharingPSO.__init__(self, True, eis[0][0], eis[0][1], eis[1][0], eis[1][1])

        if self._is_adaptive or search_and_velocity_boundaries is not None:
            # The convex boundaries need to be stored in order to apply
            # the learning strategy (using eliticism)
            # For details, see -> "Adaptive Clan Particle Swarm Optimization" ->
            # -> "III. ADAPTIVE PARTICLE SWARM OPTIMIZATION" -> "D. Learning Strategy using Elitism"
            self._spawn_boundaries = spawn_boundaries
        if search_and_velocity_boundaries is not None:
            self.__wall_type = wt

    def _find_particle_with_best_personal_best(self, greater_than: bool = False) -> Particle:
        """
        Particle that at some point had the best-known position (global min).
        Take into account the possibility that no particle is on that position when this function is called.
        This can happen when the particle which found this (i.e. the best-known) position moved to a worse position.

        :param greater_than: Returns particle with greatest position if True . Returns particle with lowest position if False (Default).

        :return: Particle with least (greater_than = False) or greatest (greater_than = True)value
        """

        # TODO: While the implementation below is computationally correct, it has one main weakness:
        #   it recalculates the fitness value for each point, which may be computationally expensive,
        #   such as in the case of the Quadric function, where the cost of computing a value is O(n^2).
        #   This principle may be generalized into other computationally expensive problems (fitness function values),
        #   such as calculating the error of a neural network training cycle.
        #   POSSIBLE SOLUTION: Store the best particle population at runtime, using either a pointer to the
        #   particle object or an index to its position in the array/list.


        # TODO: Maybe, it would be better to override operators or create a lambda function so as to utilize the standard
        # functions "min" and "max" so as to better performance.
        # Initializing it into the 1st particle for comparison.
        particle_with_best_personal_best = self.swarm[0]
        for particle in self.swarm:
            # This implementation follows the standard convention that PSO is tasked to solve a minimization problem.
            # Therefore, the default (if not only) behaviour is to consider the lower fitness value as the better solution.

            # Maximization problem.
            if greater_than:
                if particle._f_personal_best \
                        > \
                        Particle.fitness_function(particle_with_best_personal_best._personal_best_position):
                    particle_with_best_personal_best = particle
            # Minimization problem (Default).
            else:
                if particle._f_personal_best\
                        < \
                        Particle.fitness_function(particle_with_best_personal_best._personal_best_position):
                    particle_with_best_personal_best = particle

        return particle_with_best_personal_best



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
                            self.swarm.append(Particle(Particle.fitness_function, self._spawn_boundaries))
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

            # Another approach could be using the "Nearest Neighbour algorithm".
            # Note: Inherently, any attempt at making out similarity in the swarm will result in another high complexity
            # operation. Specifically, it would be O(number of particles) * O(complexity of fitness function).

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
                            self.swarm.append(Particle(self.__fitness_function, self._spawn_boundaries))
                            break


        self.__update_parameters()

        # TODO: Possibly parallelize operation (namely, position update) at particle level.
        #   Think of possibly creating two matrices by concatinating all needed information;
        #   one matrix P for positions and one matrix V for velocities and apply matrix multiplications
        #   (RV, where R would be a 3D matrix and V would be a 2D matrix containing the particles' velocities)
        #   and additions (P = P + V) respectively.
        #   Ideally, utilize GPU for the operations.

        random_multipliers = tuple(
            [
                uniform(0,1), # diag([uniform(0,1) for _ in range(len(self.__spawn_boundaries))]),  # r1
                uniform(0,1), # diag([uniform(0,1) for _ in range(len(self.__spawn_boundaries))]),  # r2
                None                                                                # r3 (potentially)
            ]
            for _ in range(len(self.swarm))
        )

        if isinstance(self.c3, float):
            for triad in random_multipliers:
                triad[2] = uniform(0,1) # diag([uniform(0,1) for _ in range(len(self.__spawn_boundaries))])

        random_multipliers_index = 0
        for particle in self.swarm:
            particle._update_position(self.global_best_position,
                                      self.w,
                                      self.c1, random_multipliers[random_multipliers_index][0],
                                      self.c2, random_multipliers[random_multipliers_index][1],
                                      self.c3, random_multipliers[random_multipliers_index][2], self._control_factor_method)
            random_multipliers_index += 1

        #? This is executed only when the Last-Elimination Principle is enabled. TODO: Why only then?
        if self.__domain_and_velocity_boundaries is not None:
            wall_enforcement()
            # limit_particle_velocity() # Velocity limitation has been moved to inside of the particle class. Doing it here is too late, as all the particles have already updated their positions.
            # replace_similar_particles()

        # Calculating the new best particle and position of the swarm.
        self._particle_with_best_personal_best = self._find_particle_with_best_personal_best()
        self.global_best_position = self._particle_with_best_personal_best._personal_best_position
        self.f_global_best = min(self.f_global_best, Particle.fitness_function(self.global_best_position))
        # print(f'gb = {self.global_best_position},\nf(gb) = {self.f_global_best}')
        # if(self._is_adaptive):
        #     print(f'ω = {self.w}, c1 = {self.c1}, c2 = {self.c2} , evolutionary state = {self._evolutionary_state}')
        # print()



        self.current_iteration += 1

    def __update_parameters(self):

        # Updating pso inertia weight (ω) and coefficients (c1, c2)
        if self._is_adaptive:
                f_evol = self._estimate_evolutionary_state()
                self._evolutionary_state = self._classify_evolutionary_state(f_evol)
                self._determine_accelaration_coefficients(self._evolutionary_state)
                self._apply_eliticism_learning_strategy(self._evolutionary_state)
                self._adapt_inertia_factor(f_evol)
        else:  # Follow classic PSO learning strategy: decrease inertia weight linearly.
            self.w = w_max - ((w_max - w_min) / self._max_iterations) * self.current_iteration


        # Updating global-local coefficient (c3) and control factor (c3_k).
        super(ClassicSwarm, self)._update_global_local_coefficient()
        super(ClassicSwarm, self)._update_global_local_control_factor()


    def calculate_swarm_distance_from_swarm_centroid(self):
        swarm_positions = [self.swarm[i]._position for i in range(len(self.swarm))]
        swarm_centroid = mean(swarm_positions, axis=0)
        # SD = √(1/N * Σ(||x_avg - x_i||^2))
        swarm_standard_deviation = sqrt(sum(norm(
            swarm_centroid - swarm_positions[i]) ** 2 for i in range(len(swarm_positions))) / len(self.swarm))
        return swarm_standard_deviation
