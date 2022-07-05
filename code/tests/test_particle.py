import unittest
from numpy import array
from classes.PSOs.particle import Particle
from scripts.benchmark_functions import sphere_function

class TestComparisonOperatorsOverloading(unittest.TestCase):

    def test_lower_than_operator(self):
        sphere_benchmark_function = sphere_function['formula']

        # The particle closest to the sphere function's minimum ( f(0,...,0) = 0 ) is particle3
        particle1 = Particle(fitness_function=sphere_benchmark_function,
                             convex_boundaries=[[-10, 10], [-10, 10]],
                             spawn_position=array([1, 1]))
        particle2 = Particle(fitness_function=sphere_benchmark_function,
                             convex_boundaries=[[-10, 10], [-10, 10]],
                             spawn_position=array([2, 2]))

        self.assertEqual(particle1 < particle2, True, 'particle1 is closer to 0 vector than particle2')

    # This is to test whether overloading all comparison operators is required or
    # whether the implementation of one the operators {lt, le, gt, ge} suffices.
    def test_greater_than_operator(self):
        sphere_benchmark_function = sphere_function['formula']

        # The particle closest to the sphere function's minimum ( f(0,...,0) = 0 ) is particle3
        particle1 = Particle(fitness_function=sphere_benchmark_function,
                             convex_boundaries=[[-10, 10], [-10, 10]],
                             spawn_position=array([1, 1]))
        particle2 = Particle(fitness_function=sphere_benchmark_function,
                             convex_boundaries=[[-10, 10], [-10, 10]],
                             spawn_position=array([2, 2]))

        self.assertEqual(particle1 > particle2, False, 'particle1 is closer to 0 vector than particle2')
