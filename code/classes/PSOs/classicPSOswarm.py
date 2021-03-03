from numpy.ma import sqrt
from classes.PSOs.particle import Particle
from random import random as r1_r2_r3_generator
from numpy import mean
from numpy.linalg import norm
from types import FunctionType


class ClassicSwarm:

    def __init__(self, swarm_or_fitness_function, convex_boundaries: list, w: float,
                 c1: float = 2, c2: float = 2, c3: float = None,
                 swarm_size: int = 50):
        if isinstance(swarm_or_fitness_function, FunctionType):
            self.swarm = [Particle(swarm_or_fitness_function, convex_boundaries) for i in range(swarm_size)]
            # self.__fitness_function = swarm_or_fitness_function
        if isinstance(swarm_or_fitness_function, list):
            self.swarm = swarm_or_fitness_function

        # Initializing the fitness_function's global best _position particle with the optimal value on spawn.
        self.global_best_position = self._find_global_best_position()
        # Storing the velocity inertia w and the learning rates c1 and c2. All three are shared among all particles.
        # Existence of constant "c3" indicates that enhanced information sharing is enabled.
        # It has the value "None" otherwise.
        self.w, self.c1, self.c2, self.c3 = w, c1, c2, c3

        # if isinstance(swarm_or_fitness_function, list):
        #     self.swarm = swarm_or_fitness_function
        #     self.global_best_position = self._find_global_best_position()

    def __find_particle_with_best_personal_best(self):
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
        r1, r2, r3 = r1_r2_r3_generator(), r1_r2_r3_generator(), None
        if self.c3 is not None: r3 = r1_r2_r3_generator()
        for particle in self.swarm:
            particle._update_position(self.global_best_position,
                                      self.w,
                                      self.c1, r1,
                                      self.c2, r2,
                                      self.c3, r3)
        self.global_best_position = self._find_global_best_position()

    def calculate_swarm_distance_from_swarm_centroid(self):
        swarm_positions = [self.swarm[i]._position for i in range(len(self.swarm))]
        swarm_centroid = mean(swarm_positions, axis=0)
        swarm_standard_deviation = sqrt(sum(norm(
            swarm_centroid - self.swarm[i]._position) ** 2 for i in range(len(self.swarm)))) / len(self.swarm)
        return swarm_standard_deviation

    def estimate_evolutionary_state(self):
        """
        Function to be used in APSO (Adaptive Particle Swarm Optimization)

        "In order to adjust properly the acceleration parameters,
        one should estimate the evolutionary state of the swarm.
        This process is divided in two separate steps.
        In the first, one needs to evaluate the average distance between each particle to all the others.
        After this, one should evaluate the the evolutionary factor (f_evol)."
        :return:
        """

        def evaluate_average_between_each_particle_to_all_others():
            d = []
            for particle1 in self.swarm:
                # "Adaptive Clan Particle Swarm Optimization" -> equation (3)
                d.append(
                    1 / (len(self.swarm) - 1) * sum(sqrt(norm(particle1._position - particle2._position))
                                                    for particle2 in self.swarm)
                )
            return d

        def evaluate_evolutionary_factor(d: list):
            best_particle_of_the_swarm = self.__find_particle_with_best_personal_best()
            d_g = 1 / (len(self.swarm) - 1) * \
                  sum(sqrt(norm(best_particle_of_the_swarm._position - particle._position))
                      for particle in self.swarm)
            d_min, d_max = min(d), max(d)

            # Note that this value (the evolutionary factor) is bounded in the values [0,1].
            return (d_g - d_min) / (d_max - d_min)

        return evaluate_evolutionary_factor(evaluate_average_between_each_particle_to_all_others())
