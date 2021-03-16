from random import random as r1_r2_generator
from random import uniform
from numpy import array as vector


class Particle:
    # w, c1, c2, c3, r1, r2, r3 = None, None, None, None, None, None, None
    fitness_function = None

    def __init__(self, fitness_function, convex_boundaries: list):
        # Randomly spawning the particle according to the problem's convex boundaries.
        self._position: vector = vector([uniform(
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
        self.__velocity: vector = vector(
            [0 for vector_space_dimension in range(len(convex_boundaries))])  # Initialize the particles as being still.

        Particle.fitness_function = fitness_function

    def _update_position(self, global_best_position: vector,
                         w: float,
                         c1: float, r1: float,
                         c2: float, r2: float,
                         c3: float, r3: float):
        if c3 is None and r3 is None:
            # Classic velocity adjustment ensues a.k.a. Enhanced Information Sharing is disabled.
            def update_velocity():
                self.__velocity = w * self.__velocity \
                                  + c1 * r1 * (self._personal_best_position - self._position) \
                                  + c2 * r2 * (global_best_position - self._position)
        else:
            # Enhanced information sharing is enabled.
            # For details see "Improved Particle Swarm Optimization Algorithm Based on
            # Last-Eliminated Principle and Enhanced Information Sharing" -> 2.2 IEPSO -> equations (2) and (3)
            def update_velocity():
                self.__velocity = w * self.__velocity \
                                  + c1 * r1 * (self._personal_best_position - self._position) \
                                  + c2 * r2 * (global_best_position - self._position) \
                                  + c3 * r3 * (global_best_position - self._personal_best_position)

        update_velocity()
        # Updating the particle's _position.
        self._position = self._position + self.__velocity
        # Checking whether the new _position is a new personal best (maximum).
        if self.__get_fitness_at_current_position() > Particle.fitness_function(self._personal_best_position):
            self._personal_best_position = self._position

    def __get_fitness_at_current_position(self):
        return Particle.fitness_function(self._position)

    def _limit_velocity(self, velocity_boundaries: list):
        # In case of false user input, (upper boundary has been given first and lower boundary second).
        # the boundaries are swapped, in order to proceed to the next part of code correctly.
        if velocity_boundaries[0] > velocity_boundaries[1]:
            velocity_boundaries[0], velocity_boundaries[1] = velocity_boundaries[1], velocity_boundaries[0]

        for i in range(len(self.__velocity)):
            if self.__velocity[i] < velocity_boundaries[0]:
                self.__velocity[i] = velocity_boundaries[0]
            if self.__velocity[i] > velocity_boundaries[1]:
                self.__velocity[i] = velocity_boundaries[1]

