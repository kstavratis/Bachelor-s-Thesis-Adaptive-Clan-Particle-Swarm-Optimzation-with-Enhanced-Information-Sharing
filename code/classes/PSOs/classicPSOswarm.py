from numpy.ma import sqrt
from random import random as r1_r2_r3_generator, uniform, randrange, gauss
from numpy import mean, e
from numpy.linalg import norm
from types import FunctionType
from enum import Enum

from classes.PSOs.particle import Particle
from classes.evolutionary_states import EvolutionaryStates
from scripts.evolutionary_state_classification import classify_evolutionary_state

w_min, w_max = 0.4, 0.9
c_min, c_max = 1.5, 2.5




class ClassicSwarm:
    def __init__(self, swarm_or_fitness_function, convex_boundaries: list, maximum_iterations: int,
                 w: float,
                 adaptive: bool = False,
                 c1: float = 2, c2: float = 2, c3: float = None,
                 swarm_size: int = 50):
        if isinstance(swarm_or_fitness_function, FunctionType):
            self.swarm = [Particle(swarm_or_fitness_function, convex_boundaries) for i in range(swarm_size)]
        if isinstance(swarm_or_fitness_function, list):
            self.swarm = swarm_or_fitness_function
        if adaptive:
            # The convex boundaries need to be stored in order to apply
            # the learning strategy (using eliticism)
            # For details, see -> "Adaptive Clan Particle Swarm Optimization" ->
            # -> "III. ADAPTIVE PARTICLE SWARM OPTIMIZATION" -> "D. Learning Strategy using Elitism"
            self.__convex_boundaries = convex_boundaries

        # Initializing the fitness_function's global best _position particle with the optimal value on spawn.
        self.__fitness_function = swarm_or_fitness_function
        self.global_best_position = self._find_global_best_position()
        # Storing the velocity inertia w and the learning rates c1, c2 & c3.
        # All three are shared among all particles of the swarm.
        # Existence of constant c3 indicates that enhanced information sharing is enabled.
        self.w, self.c1, self.c2, self.c3 = w, c1, c2, c3

        self.__adaptive = adaptive
        self.__max_iterations, self.__current_iteration = maximum_iterations, 0

    def __find_particle_with_best_personal_best(self) -> Particle:
        """
        Particle that at some point had the best-known position (global max).
        Take into account the possibility that no particle is on that position when this function is called.
        This can happen when the particle which found this position moved to a worse position.
        :return:
        """
        # Initializing it into the 1st particle for comparison.
        particle_with_best_personal_best = self.swarm[0]
        for particle in self.swarm:
            if Particle.fitness_function(particle._personal_best_position) \
                    > \
                    Particle.fitness_function(particle_with_best_personal_best._personal_best_position):
                particle_with_best_personal_best = particle

        return particle_with_best_personal_best

    def _find_global_best_position(self):
        """
        Best position that the swarm has found during execution of the algorithm.
        :return:
        """
        return self.__find_particle_with_best_personal_best()._personal_best_position

    def update_swarm(self):
        self.__update_parameters()

        r1, r2, r3 = r1_r2_r3_generator(), r1_r2_r3_generator(), None
        if self.c3 is not None: r3 = r1_r2_r3_generator()
        for particle in self.swarm:
            particle._update_position(self.global_best_position,
                                      self.w,
                                      self.c1, r1,
                                      self.c2, r2,
                                      self.c3, r3)
        self.global_best_position = self._find_global_best_position()

        self.__current_iteration += 1

    def __update_parameters(self):
        if self.__adaptive:
            self.__determine_accelaration_coefficients()
            self.__apply_eliticism_learning_strategy()
            self.__adapt_inertia_factor()
        else:  # Follow classic PSO learning strategy: decrease inertia weight linearly.
            self.w = w_max - ((w_max - w_min) / self.__max_iterations) * self.__current_iteration

    def calculate_swarm_distance_from_swarm_centroid(self):
        swarm_positions = [self.swarm[i]._position for i in range(len(self.swarm))]
        swarm_centroid = mean(swarm_positions, axis=0)
        swarm_standard_deviation = sqrt(sum(norm(
            swarm_centroid - self.swarm[i]._position) ** 2 for i in range(len(self.swarm)))) / len(self.swarm)
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
            for particle1 in self.swarm:
                # "Adaptive Clan Particle Swarm Optimization" -> equation (3)
                d.append(
                    1 / (len(self.swarm) - 1) * sum(norm(particle1._position - particle2._position)
                                                    for particle2 in self.swarm)
                )
            return d

        def evaluate_evolutionary_factor(d: list):
            best_particle_of_the_swarm = self.__find_particle_with_best_personal_best()
            d_g = 1 / (len(self.swarm) - 1) *\
                  sum(norm(best_particle_of_the_swarm._position - particle._position)
                      for particle in self.swarm)
            d_min, d_max = min(d), max(d)

            # Note that this value (the evolutionary factor) is bounded in the values [0,1].
            return (d_g - d_min) / (d_max - d_min)

        return evaluate_evolutionary_factor(evaluate_average_between_each_particle_to_all_others())

    def __determine_accelaration_coefficients(self):

        class CoefficientOperations(Enum):
            INCREASE = 0,
            DECREASE = 1,
            INCREASE_SLIGHTLY = 2,
            DECREASE_SLIGHTLY = 3

        acceleration_rate = uniform(0.05, 0.10)
        c1_strategy, c2_strategy = "", ""
        f_evol = self.__estimate_evolutionary_state()
        evolutionary_state = classify_evolutionary_state(f_evol)
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
            raise ValueError  # The evolutionary state can only be in one of the four states above.


        if c1_strategy == CoefficientOperations.INCREASE:
            self.c1 += acceleration_rate
        elif c1_strategy == CoefficientOperations.INCREASE_SLIGHTLY:
            self.c1 += 0.5 * acceleration_rate
        elif c1_strategy == CoefficientOperations.DECREASE:
            self.c1 -= acceleration_rate

        if c2_strategy == CoefficientOperations.INCREASE:
            self.c2 += acceleration_rate
        elif c2_strategy == CoefficientOperations.INCREASE_SLIGHTLY:
            self.c2 += 0.5 * acceleration_rate
        elif c2_strategy == CoefficientOperations.DECREASE:
            self.c2 -= acceleration_rate
        elif c2_strategy == CoefficientOperations.DECREASE_SLIGHTLY:
            self.c2 -= 0.5 * acceleration_rate


        print("Evolutionary state = " + str(evolutionary_state))


        # In order to avoid an explosion state, it is necessary to bound the sum c1+c2.
        # As the minimum and the maximum value for c1 and c2 are c_min = 1.5 and c_max = 2.5,
        # when c1 + c2 > cm + cmax, one should update c1 and c2 using:
        # c_i = c_i/(c1 + c2) (c_min + c_max):
        if sum([self.c1, self.c2]) > c_min + c_max:
            c1_old, c2_old = self.c1, self.c2
            self.c1 = c1_old / (c1_old + c2_old) * (c_min + c_max)
            self.c2 = c2_old / (c1_old + c2_old) * (c_min + c_max)

    def __apply_eliticism_learning_strategy(self):
        # This strategy aims at maintaining the diversity during the search,
        # by causing a escape state of the best particle when it gets trapped in a local minimum.
        # That is why this strategy is executed only in case of convergence.
        if self.__estimate_evolutionary_state() == EvolutionaryStates.CONVERGENCE:
            search_space_dimention = randrange(len(self.__convex_boundaries))
            mu = mean(self.__convex_boundaries[search_space_dimention])
            # "Adaptive Particle Swarm Optimization, Zhi et al." -> "IV. APSO" -> "C. ELS" ->
            # "Empirical study shows that σ_max = 1.0 and σ_min = 0.1
            # result in good performance on most of the test functions"
            sigma = 1 - (1 - 0.1) / self.__max_iterations * self.__current_iteration
            elitist_learning_rate = gauss(mu, sigma)

            self.__find_particle_with_best_personal_best()._position[search_space_dimention] += \
                elitist_learning_rate

    def __adapt_inertia_factor(self):
        f_evol = self.__estimate_evolutionary_state()
        self.w = 1 / (1 + 1.5 * e ** (-2.6 * f_evol))  # ∈[0.4, 0.9]  ∀f_evol ∈[0,1]
