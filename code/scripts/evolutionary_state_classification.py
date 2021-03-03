from numpy import inf

from classes.evolutionary_states import EvolutionaryStates

"""
For details see "Adaptive Particle Swarm Optimization (Zhan et al.)" -> III ESE for PSO -> B. ESE -> Step 3
The singleton defuzzification classification technique is used to determine the state the swarm is in.
"""


def classify_evolutionary_state(evolutionary_factor: float):
    def exploration_membership_function(f: float):
        if 0 <= f <= 0.4 or 0.8 < f <= 1.0:
            return 0
        elif 0.4 < f <= 0.6:
            return 5 * f - 2
        elif 0.6 < f <= 0.7:
            return 1
        elif 0.7 < f <= 0.8:
            return -10 * f + 8
        else:
            raise ValueError  # The evolutionary factor is bounded in the values [0,1].

    def exploitation_membership_function(f: float):
        if 0 <= f <= 0.2 or 0.6 < f <= 1:
            return 0
        elif 0.2 < f <= 0.3:
            return 10 * f - 2
        elif 0.3 < f <= 0.4:
            return 1
        elif 0.4 < f <= 0.6:
            return -5 * f + 3
        else:
            raise ValueError  # The evolutionary factor is bounded in the values [0,1].

    def convergence_membership_function(f: float):
        if 0 <= f <= 0.1:
            return 1
        elif 0.1 < f <= 0.3:
            return -5 * f + 1.5
        elif 0.3 < f <= 1:
            return 0
        else:
            raise ValueError  # The evolutionary factor is bounded in the values [0,1].

    def jumping_out_membership_function(f: float):
        if 0 <= f <= 0.7:
            return 0
        elif 0.7 < f <= 0.9:
            return 5 * f - 3.5
        elif 0.9 < f <= 1:
            return 1
        else:
            raise ValueError  # The evolutionary factor is bounded in the values [0,1].

    membership_function_dict = {
        EvolutionaryStates.EXPLORATION: exploration_membership_function,
        EvolutionaryStates.EXPLOITATION: exploitation_membership_function,
        EvolutionaryStates.CONVERGENCE: convergence_membership_function,
        EvolutionaryStates.JUMP_OUT: jumping_out_membership_function
    }

    f = [-inf for i in range(len(EvolutionaryStates))]
    f[EvolutionaryStates.EXPLORATION.value] = \
        membership_function_dict[EvolutionaryStates.EXPLORATION](evolutionary_factor)

    f[EvolutionaryStates.EXPLOITATION.value] = \
        membership_function_dict[EvolutionaryStates.EXPLOITATION](evolutionary_factor)

    f[EvolutionaryStates.CONVERGENCE.value] = \
        membership_function_dict[EvolutionaryStates.CONVERGENCE](evolutionary_factor)

    f[EvolutionaryStates.JUMP_OUT.value] = \
        membership_function_dict[EvolutionaryStates.JUMP_OUT](evolutionary_factor)

    return EvolutionaryStates(f.index(max(f)))  # singleton defuzzification classification technique
