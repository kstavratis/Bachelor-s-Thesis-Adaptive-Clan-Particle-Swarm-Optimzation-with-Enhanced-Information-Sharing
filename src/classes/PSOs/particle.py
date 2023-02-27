"""
Copyright (C) 2022  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from random import uniform

import numpy
from numpy import array as vector, zeros, absolute, ndarray, dot

from classes.PSOs.Mixins.enhanced_information_sharing.enums.control_factor_types import ControlFactorTypes


class Particle:
    # w, c1, c2, c3, r1, r2, r3 = None, None, None, None, None, None, None
    fitness_function = None
    convex_boundaries = None
    velocity_boundaries = None

    def __init__(self, fitness_function, convex_boundaries: list, spawn_position: vector = None):

        # Particle common attributes
        # --------------------------
        # 1) "Convex boundaries"    := The subspace of R^n which the particle is allowed to move in.
        # 2) "Velocity boundaries"  := The minimum and maximum velocity each particle is allowed to reach.
        # 3) "Objective function"   := Particles of the swarm "live" inside the same "environment".

        # In case of false user input, (upper boundary has been given first and lower boundary second).
        # the boundaries are swapped, in order to proceed to the next part of code correctly.
        for dimension in range(len(convex_boundaries)):
            if convex_boundaries[dimension][0] > convex_boundaries[dimension][1]:
                convex_boundaries[dimension][0], convex_boundaries[dimension][1] = convex_boundaries[dimension][1], convex_boundaries[dimension][0]
        Particle.convex_boundaries = convex_boundaries
        # Note: It is good practice to have the speed limits be a multiple of the domain limits.
        # In the article "Adaptive Particle Swarm Optimization", Zhan et al. (https://ieeexplore.ieee.org/document/4812104)
        # used a Vmax of 20% the domain limits in each dimension.
        # Adaptive Particle Swarm Optimization -> II. PSO AND ITS DEVELOPMENTS -> A. PSO Framework
        # Desired limits: # −0.2max{ |𝑋𝑚𝑖𝑛𝑑|, |𝑋𝑚𝑎𝑥𝑑|}) and 0.2max{|𝑋𝑚𝑖𝑛𝑑|,|𝑋𝑚𝑎𝑥𝑑|}, ∀d∈D (D===problem domain)
        # Note: There is no need to check for false user input, as (currently) the velocity boundaries are extracted
        # by the convex boundaries, which have already been corrected above.
        velocity_boundaries = vector(Particle.convex_boundaries); velocity_boundaries = 0.2 * abs(velocity_boundaries)
        for dimension_limits in velocity_boundaries:
            dimension_limits[0] = -dimension_limits[0]
        Particle.velocity_boundaries = velocity_boundaries # Setting the velocity boundaries swarm-wise, utilizing a shared variable. This is to reduce computational load in "__limit_velocity()".
        Particle.fitness_function = fitness_function


        # Individual particle attributes
        # ------------------------------
        # 1) Position
        # 2) Velocity
        # 3) Memory

        # 1) Position
        # Randomly spawning the particle according to the problem's convex boundaries.
        if spawn_position is None:
            self._position: vector = vector([uniform(
                convex_boundaries[vector_space_dimension][0], convex_boundaries[vector_space_dimension][1])
                for vector_space_dimension in range(len(convex_boundaries))])
        if isinstance(spawn_position, ndarray):
            self._position: vector = spawn_position

        # 2) Velocity
        # TODO
        #  1)Verify whether the velocities will have boundaries or not.
        #  Will depend on algorithm used, e.g. Last-Eliminated Principle PSO.
        #  2) At which speeds is it a good idea to start the algorithm? With small initial speeds, the particles
        #  will have a larger velocity, tending at the global optimum
        # self.__velocity: vector = vector([particle_position_and_velocity_initializer(0, 1)
        #                                  for vector_space_dimension in range(len(spawn_boundaries))])
        self.__velocity: vector = zeros(len(convex_boundaries))  # Initialize the particles as being still.

        # 3) Memory
        # The particle's first personal best _position is the _position where it is spawned.
        self._personal_best_position: vector = self._position
        # Auxiliary variable which stores the evaluation of the objective function at the particle's personal best position.
        # This variable serves to bring speed-up to the algorithm.
        self._f_personal_best = Particle.fitness_function(self._personal_best_position)




    def __lt__(self, other):
        return Particle.fitness_function(self._position) < Particle.fitness_function(other._position)

    def _update_position(self, global_best_position: vector,
                         w: float,
                         c1: float, r1: float or numpy.array,
                         c2: float, r2: float or numpy.array,
                         c3: float, r3: float or numpy.array, c3_k_control_factor_mode: ControlFactorTypes):
        if c3 is None and r3 is None:
            # Classic velocity adjustment ensues a.k.a. Enhanced Information Sharing is disabled.
            def update_velocity():
                self.__velocity = w * self.__velocity \
                                  + c1 * dot(r1, (self._personal_best_position - self._position)) \
                                  + c2 * dot(r2, (global_best_position - self._position))
        else:
            # Enhanced information sharing is enabled.
            # For details see "Improved Particle Swarm Optimization Algorithm Based on
            # Last-Eliminated Principle and Enhanced Information Sharing" -> 2.2 IEPSO -> equations (2) and (3)
            def update_velocity():

                if c3_k_control_factor_mode != ControlFactorTypes.ADAPTIVE:
                    # Follow article's approach.
                    phi3 = c3 * dot(r3, (absolute(global_best_position - self._personal_best_position)))
                else:
                    # If an adaptive strategy is used, show bias towards:
                    # - global, if state = {CONVERGENCE, JUMP-OUT}
                    # - local, if state = {EXPLORATION. EXPLOITATION}
                    # NOTE: There is no need to create an if-clause to change the bias/direction of the third
                    # component. That is because the change in direction ALREADY takes place when the sign
                    # of the c3_k control factor adapts (positive for {CONVERGENCE, JUMP-OUT} and negative
                    # for {EXPLORATION, EXPLOITATION})
                    phi3 = c3 * dot(r3, (global_best_position - self._personal_best_position))


                self.__velocity = w * self.__velocity \
                                  + c1 * dot(r1, (self._personal_best_position - self._position)) \
                                  + c2 * dot(r2, (global_best_position - self._position)) \
                                  + phi3


        update_velocity()
        self.__limit_velocity(Particle.velocity_boundaries)
        # Updating the particle's _position.
        self._position = self._position + self.__velocity
        # "Glue" the particle whenever it would exceed the search boundaries, instead of letting it move freely.
        self.__bind_particle_to_convex_boundaries()
        # Checking whether the new _position is a new personal best (minimum).
        #! PERFORMANCE COMMENT
        #! One additional evaluation of the objective function is made for each particle at each iteration!
        # TODO: Move this portion of the code to a higher level (the swarm), where the evaluations already take place.
        f_current = Particle.fitness_function(self._position)
        if f_current < self._f_personal_best:
            self._personal_best_position = self._position
            self._f_personal_best = f_current
        # TODO: Should the best global particle be updated at this point as well or should there be a wait for all
        #   particles to complete their iteration? According to the pseudocode in Wikipedia (), the former approach
        #   is implied.
        # CURRENTLY, THE ABOVE "2DO" IS EXPLORED IN A DIFFERENT BRANCH.


    def __limit_velocity(self, velocity_boundaries: list):
        for dimension in range(len(velocity_boundaries)):
            if self.__velocity[dimension] < velocity_boundaries[dimension][0]:
                self.__velocity[dimension] = velocity_boundaries[dimension][0]
            if self.__velocity[dimension] > velocity_boundaries[dimension][1]:
                self.__velocity[dimension] = velocity_boundaries[dimension][1]

    def __bind_particle_to_convex_boundaries(self):
        for dimension in range(len(self._position)):
            if self._position[dimension] < Particle.convex_boundaries[dimension][0]:
                self._position[dimension] = Particle.convex_boundaries[dimension][0]
            if self._position[dimension] > Particle.convex_boundaries[dimension][1]:
                self._position[dimension] = Particle.convex_boundaries[dimension][1]

    def _zero_velocity(self, axis_index: int):
        self.__velocity[axis_index] = 0

    def _reflect_velocity(self, axis_index: int):
        self.__velocity[axis_index] *= -1

