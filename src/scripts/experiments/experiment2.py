"""
Copyright (C) 2022  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

import pandas as pd
from typing import Any
from numpy import inf
from scipy.linalg import norm


from classes.PSOs.classicPSOswarm import ClassicSwarm
from classes.PSOs.clanPSOswarm import ClanSwarm

# This strict policy is enacted so as to catch rounding errors (overflows/underflows) as well.
import warnings
warnings.filterwarnings("error")

loop_stop_condition_limit = 5e-15


def experiment(objective_function_pointer: Any,
                swarm_size: int = 40,
                is_clan: bool = False,
                number_of_clans: int = 4,
                kwargs: dict = {}
               ):
    """

    """

    objective_function_goal_point = kwargs["objective_function_goal_point"]; del kwargs["objective_function_goal_point"]

    if not is_clan:
        experiment_swarm = ClassicSwarm(swarm_or_fitness_function=objective_function_pointer,
                                        swarm_size=swarm_size,
                                        **kwargs)

    if is_clan:
        experiment_swarm = ClanSwarm(fitness_function=objective_function_pointer,
                                     swarm_size=swarm_size // number_of_clans, number_of_clans=number_of_clans,
                                     **kwargs)

    
    # START EXPERIMENT

    iteration = 0
    maximum_iterations = kwargs["maximum_iterations"]
    loop_stop_condition_value = inf

    data_dictionary = dict()
    iteration_list = []
    distance_from_target_list = []
    

    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration <= maximum_iterations:

        # Store iteration result.
        iteration_list.append(iteration)
        if not is_clan:
            distance_from_target_list.append(norm(experiment_swarm.global_best_position - objective_function_goal_point))
        elif is_clan:
            distance_from_target_list.append(norm(experiment_swarm.find_population_global_best_position() - objective_function_goal_point))

        # Execute an iteration of the algorithm.
        try:
            experiment_swarm.update_swarm()
        # Stopping process when rounding errors (overflow or underflow) appear.
        except FloatingPointError:
            iteration += 1
            break
        except RuntimeWarning:
            iteration += 1
            break

        # Updating termination-decision variables.
        loop_stop_condition_value = experiment_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
        
    # END EXPERIMENT



    data_dictionary = {
        "Iteration" : iteration_list,
        "Centroid distance" : distance_from_target_list
    }

    df = pd.DataFrame(data_dictionary)

    return df