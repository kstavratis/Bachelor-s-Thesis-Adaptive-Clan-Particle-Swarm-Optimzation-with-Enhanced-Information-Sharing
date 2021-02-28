from numpy.ma import sqrt
from .particle import Particle
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
            # Initializing the fitness_function's global best _position particle with the optimal value on spawn.
            self.__fitness_function = swarm_or_fitness_function
            self.global_best_position = self._find_global_best_position()
            # Storing the velocity inertia w and the learning rates c1 and c2. All three are shared among all particles.
            Particle.w, Particle.c1, Particle.c2 = w, c1, c2
            if c3 is not None:  # Existence of constant c3 indicates that enhanced information sharing is enabled.
                Particle.c3 = c3
        if isinstance(swarm_or_fitness_function, list):
            self.swarm = swarm_or_fitness_function
            self.global_best_position = self._find_global_best_position()

    def _find_global_best_position(self):
        # Initializing it into the 1st particle for comparison.
        global_best_position = self.swarm[0]._personal_best_position
        for particle in self.swarm:
            if Particle.fitness_function(particle._personal_best_position) \
                    > \
                    Particle.fitness_function(global_best_position):
                global_best_position = particle._personal_best_position
        return global_best_position

    def update_swarm(self):
        Particle.r1, Particle.r2 = r1_r2_r3_generator(), r1_r2_r3_generator()
        if Particle.c3 is not None: Particle.r3 = r1_r2_r3_generator()
        for particle in self.swarm:
            particle._update_position(self.global_best_position)
        self.global_best_position = self._find_global_best_position()

    def calculate_swarm_distance_from_swarm_centroid(self):
        swarm_positions = [self.swarm[i]._position for i in range(len(self.swarm))]
        swarm_centroid = mean(swarm_positions, axis=0)
        swarm_standard_deviation = sqrt(sum(norm(
            swarm_centroid - self.swarm[i]._position) ** 2 for i in range(len(self.swarm)))) / len(self.swarm)
        return swarm_standard_deviation
