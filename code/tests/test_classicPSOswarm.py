import unittest
from numpy import array
from classes.PSOs.particle import Particle
from scripts.benchmark_functions import sphere_function
from classes.PSOs.classicPSOswarm import ClassicSwarm

class TestPSOSolvesMinimizationProblem(unittest.TestCase):

    def test_find_particle_with_best_personal_best(self):
        sphere_benchmark_function = sphere_function['formula']

        # The particle closest to the sphere function's minimum ( f(0,...,0) = 0 ) is particle3
        particle1 = Particle(fitness_function=sphere_benchmark_function,
                             convex_boundaries=[[-10, 10], [-10, 10]],
                             spawn_position=array([1, 1]))
        particle2 = Particle(fitness_function=sphere_benchmark_function,
                             convex_boundaries=[[-10, 10], [-10, 10]],
                             spawn_position=array([2, 2]))
        particle3 = Particle(fitness_function=sphere_benchmark_function,
                             convex_boundaries=[[-10, 10], [-10, 10]],
                             spawn_position=array([0.5, 0.5]))

        test_swarm = ClassicSwarm(swarm_or_fitness_function=[particle1, particle2, particle3],
                                  spawn_boundaries=[[-100, 100],[-100, 100]],
                                  maximum_iterations=5000)

        test_best_particle = test_swarm._ClassicSwarm__find_particle_with_best_personal_best()

        self.assertEqual(first=test_best_particle,
                         second=particle3,
                         msg=f'Expected best particle was "particle 3", with position = {particle3._position}')

