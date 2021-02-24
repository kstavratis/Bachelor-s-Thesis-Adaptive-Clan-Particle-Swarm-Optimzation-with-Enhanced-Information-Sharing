from random import random as r1_r2_generator
from random import uniform as particle_position_and_velocity_initializer
from numpy import array as vector


class Particle:
    w, c1, c2, r1, r2 = None, None, None, None, None
    fitness_function = None

    def __init__(self, fitness_function, convex_boundaries: list):
        # Randomly spawning the particle according to the problem's convex boundaries.
        self._position: vector = vector([particle_position_and_velocity_initializer(
            convex_boundaries[vector_space_dimension][0], convex_boundaries[vector_space_dimension][1])
            for vector_space_dimension in range(len(convex_boundaries))])
        # The particle's first personal best _position is the _position where it is spawned.
        self._personal_best_position: vector = self._position

        # TODO
        #  1)Verify whether the velocities will have boundaries or not.
        #  Will depend on algorithm used, e.g. Last-Eliminated Principle PSO.
        #  2) At which speeds is it a good idea to start the algorithm? With small initial speeds, the particles
        #  will have a larger velocity, tending at the global optimum
        # self.__velocity: vector = vector([particle_position_and_velocity_initializer(0, 1)
        #                                  for vector_space_dimension in range(len(convex_boundaries))])
        self.__velocity: vector = vector([0 for vector_space_dimension in range(len(convex_boundaries))]) # Initialize the particles as being still.

        Particle.fitness_function = fitness_function

    def _update_position(self, global_best_position: vector):
        def update_velocity():
            # Note: global_best_position is calculated in class "ClassicSwarm" -> "_find_global_best_position" function.
            self.__velocity = self.w * self.__velocity \
                              + self.c1 * self.r1 * (self._personal_best_position - self._position) \
                              + self.c2 * self.r2 * (global_best_position - self._position)

        update_velocity()
        # Updating the particle's _position.
        self._position = self._position + self.__velocity
        # Checking whether the new _position is a new personal best (maximum).
        if self.__get_fitness_at_current_position() > Particle.fitness_function(self._personal_best_position):
            self._personal_best_position = self._position

    def __get_fitness_at_current_position(self):
        return Particle.fitness_function(self._position)

