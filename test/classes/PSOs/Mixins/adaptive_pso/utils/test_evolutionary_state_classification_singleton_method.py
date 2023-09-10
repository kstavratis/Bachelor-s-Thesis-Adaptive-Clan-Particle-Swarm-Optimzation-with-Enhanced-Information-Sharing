"""
A file which supports the validity of the `evolutionary_state_classification_singleton_method.py`
script is implemented correctly by providing normal-case and edge-case use of all possible outcomes.


Copyright (C) 2023  Konstantinos Stavratis
"""

# Append the root directory of the project, so as to allow Python
# to correctly import the required (custom) modules.
import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = root_dir[:root_dir.index('test')]
sys.path.append(root_dir)

import src.classes.PSOs.Mixins.adaptive_pso.utils.evolutionary_state_classification_singleton_method as clasfation
from src.classes.PSOs.Mixins.adaptive_pso.enums.evolutionary_states import EvolutionaryStates


# Test case definitions start here.
import unittest

class TestEvolutionaryStateClassificationBoundaries(unittest.TestCase):

    def test_convergence_state_outside_of_left_boundary(self):
        with self.assertRaises(ValueError):
            clasfation.classify_evolutionary_state(evolutionary_factor=-10)

    def test_convergence_state_just_outside_of_left_boundary(self):
        with self.assertRaises(ValueError):
            clasfation.classify_evolutionary_state(evolutionary_factor=1e-9)

    def test_convergence_state_on_left_boundary(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0)
        self.assertEqual(evolutionary_state, EvolutionaryStates.CONVERGENCE)

    def test_convergence_state_near_left_boundary(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor =1e-9)
        self.assertEqual(evolutionary_state, EvolutionaryStates.CONVERGENCE)

    def test_convergence_state_normal_case(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.1)
        self.assertEqual(evolutionary_state, EvolutionaryStates.CONVERGENCE)




    # CONVERGENCE and EXPLOITATION membership functions are both active in the interval [0.2, 0.3]

    def test_convergence_and_exploitation_common_area_convergence_wins(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.21)
        self.assertTrue(evolutionary_state == EvolutionaryStates.CONVERGENCE or
                        evolutionary_state == EvolutionaryStates.EXPLOITATION)

    def test_convergence_and_exploitation_intersection(self):
        # y = y <=> -5f + 1.5 = 10f - 2 <=> 15f = 3.5 <=> f = 3.5/15 = 0.7/3
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.7 / 3)
        self.assertTrue(evolutionary_state == EvolutionaryStates.CONVERGENCE or
                        evolutionary_state == EvolutionaryStates.EXPLOITATION)

    def test_convergence_and_exploitation_common_area_exploitation_wins(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.29)
        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLOITATION)

    def test_exploitation_normal_case(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.35)
        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLOITATION)




    # EXPLOITATION and EXPLORATION membership functions are both active in the interval [0.4, 0.6]

    def test_exploitation_and_exploration_common_area_exploitation_wins(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.45)
        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLOITATION)

    def test_exploitation_and_exploration_intersection(self):
        # y = y <=> -5f - 2 = -5f + 3 <=> 10f = 5 <=> f = 1/2 = 0.5
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.5)
        self.assertTrue(evolutionary_state == EvolutionaryStates.EXPLOITATION or
                        evolutionary_state == EvolutionaryStates.EXPLORATION)

    def test_exploitation_and_exploration_common_area_exploration_wins(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.55)
        self.assertTrue(evolutionary_state == EvolutionaryStates.EXPLORATION or
                        evolutionary_state == EvolutionaryStates.EXPLOITATION)

    def test_exploration_normal_case(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.65)
        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLORATION)



    # EXPLORATION and JUMP_OUT membership functions are both active in the interval [0.7, 0.8]

    def test_exploration_and_jumpout_common_area_exploration_wins(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.72)
        self.assertTrue(evolutionary_state == EvolutionaryStates.EXPLORATION or
                        evolutionary_state == EvolutionaryStates.JUMP_OUT)

    def test_exploration_and_jumpout_intersection(self):
        # y = y <=> 5f - 3.5 = -10f + 8 <=> 15f = 11.5 <=> f = 11.5/15 = 2.3/3
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=2.3 / 3)
        self.assertTrue(evolutionary_state == EvolutionaryStates.EXPLORATION or
                        evolutionary_state == EvolutionaryStates.JUMP_OUT)

    def test_exploration_and_jumpout_common_area_jumpout_wins(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.78)
        self.assertEqual(evolutionary_state, EvolutionaryStates.JUMP_OUT)

    def test_jumpout_normal_case(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=0.9)
        self.assertEqual(evolutionary_state, EvolutionaryStates.JUMP_OUT)


    def test_jumpout_state_near_left_boundary(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=1 - 10 ** (-9))
        self.assertEqual(evolutionary_state, EvolutionaryStates.JUMP_OUT)

    def test_jumpout_state_on_right_boundary(self):
        evolutionary_state = clasfation.classify_evolutionary_state(evolutionary_factor=1)
        self.assertEqual(evolutionary_state, EvolutionaryStates.JUMP_OUT)

    def test_jumpout_state_just_outside_of_right_boundary(self):
        with self.assertRaises(ValueError):
            clasfation.classify_evolutionary_state(evolutionary_factor=1 + 10 ** (-9))

    def test_jumpout_state_outside_of_right_boundary(self):
        with self.assertRaises(ValueError):
            clasfation.classify_evolutionary_state(evolutionary_factor=1.5)







class TestEvolutionaryStateClassificationConflicts(unittest.TestCase):
    def test_exploration_vs_exploitation_previous_exploration(self):
        clasfation.current_state = EvolutionaryStates.EXPLORATION
        evolutionary_state = clasfation.classify_evolutionary_state(0.5)

        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLORATION)

    def test_exploration_vs_exploitation_previous_exploitation(self):
        clasfation.current_state = EvolutionaryStates.EXPLOITATION
        evolutionary_state = clasfation.classify_evolutionary_state(0.5)

        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLOITATION)

    def test_exploration_vs_exploitation_previous_convergence(self):
        clasfation.current_state = EvolutionaryStates.CONVERGENCE
        evolutionary_state = clasfation.classify_evolutionary_state(0.5)

        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLOITATION)

    def test_exploration_vs_exploitation_previous_jump_out(self):
        clasfation.current_state = EvolutionaryStates.JUMP_OUT
        evolutionary_state = clasfation.classify_evolutionary_state(0.5)

        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLORATION)

    def test_exploration_vs_jump_out_previous_exploration(self):
        clasfation.current_state = EvolutionaryStates.EXPLORATION
        evolutionary_state = clasfation.classify_evolutionary_state(0.75)

        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLORATION)

    def test_exploration_vs_jump_out_previous_exploitation(self):
        clasfation.current_state = EvolutionaryStates.EXPLOITATION
        evolutionary_state = clasfation.classify_evolutionary_state(0.75)

        self.assertEqual(evolutionary_state, EvolutionaryStates.JUMP_OUT)

    def test_exploration_vs_jump_out_previous_convergence(self):
        clasfation.current_state = EvolutionaryStates.CONVERGENCE
        evolutionary_state = clasfation.classify_evolutionary_state(0.75)

        self.assertEqual(evolutionary_state, EvolutionaryStates.JUMP_OUT)

    def test_exploration_vs_jump_out_previous_jump_out(self):
        clasfation.current_state = EvolutionaryStates.JUMP_OUT
        evolutionary_state = clasfation.classify_evolutionary_state(0.75)

        self.assertEqual(evolutionary_state, EvolutionaryStates.JUMP_OUT)

    def test_exploitation_vs_convergence_previous_exploration(self):
        clasfation.current_state = EvolutionaryStates.EXPLORATION
        evolutionary_state = clasfation.classify_evolutionary_state(0.25)

        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLOITATION)

    def test_exploitation_vs_convergence_previous_exploitation(self):
        clasfation.current_state = EvolutionaryStates.EXPLOITATION
        evolutionary_state = clasfation.classify_evolutionary_state(0.25)

        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLOITATION)

    def test_exploitation_vs_convergence_previous_convergence(self):
        clasfation.current_state = EvolutionaryStates.CONVERGENCE
        evolutionary_state = clasfation.classify_evolutionary_state(0.25)

        self.assertEqual(evolutionary_state, EvolutionaryStates.CONVERGENCE)

    def test_exploitation_vs_convergence_previous_jump_out(self):
        clasfation.current_state = EvolutionaryStates.JUMP_OUT
        evolutionary_state = clasfation.classify_evolutionary_state(0.25)

        self.assertEqual(evolutionary_state, EvolutionaryStates.EXPLOITATION)