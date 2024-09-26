"""
This file is responsible for the main loop of Particle Swarm Optimization (PSO).

Copyright (C) 2024  Konstantinos Stavratis
e-mail: kostauratis@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from src.scripts.io_handling.config_initialization_handler import handle_config_file_data
from src.scripts.io_handling.data_generation.extractor import swarm_positions_extractor, gbest_position_extractor
from src.scripts.io_handling.data_generation.logger import log_pso
from src.scripts.benchmark_functions import benchmark_functions as bf 

from src.classes.PSOs.pso_backbone import PSOBackbone
from src.classes.PSOs.clan_pso import ClanPSO
from src.classes.PSOs.utils import positions_initializers

from ..io_handling.data_generation.logger import log_pso

import numpy as np
import pandas as pd


def run(data : dict, seed: int):
    """
    Parameters
    ----------
    data : dict
        This parameter's elements contain the details of the conducted experiment.
        The minimum such details required are 'max_iterations', 'objective_function', 'nr_dimensions'

    seed : int
        The seed ensures reproducability of experiments in cases where benchmarking is of importance.
        Currently, the seed is utilized only for controlling the particles' initial positions.
    """
    # Hyperparameter declarations
    maximum_iterations = None
    objective_function = None

    # Variable declarations
    current_error, numerical_precision_termination_criterion = None , 1e-14

    maximum_iterations = data['max_iterations']
    bf.domain_dimensions = data['nr_dimensions'] ; bf.reset_benchmark_functions_dictionaries()
    # The benchmark functions are reset, so as to account for the change in the number of dimensions
    # (e.g. `bf.function_name['goal_point']`, `bf.function_name['search_domain']` ) 
    objective_function = getattr(bf, data['objective_function'])
    #* Order between the last two (physical) lines matter!
    #* The benchmark functions must be reset first
    #* before assigning the result into the local variable `objective_function`


    pso_instance = handle_config_file_data(data)
    
    # Set the initial positions of the swarm(s).
    # This is mainly to account for experiments which we wish to have some control over
    # (e.g. identical position initializations of experiments or reproducability of results).
    # If no control is required, then this does not change anything w.r.t. to the algorithm;
    # with the current implementation, where the swarms are initialized internally,
    # applying the following lines of code is identical to initializing twice,
    # which, in a random setting like this, does not change anything.
    if isinstance(pso_instance, PSOBackbone):
        pso_instance = positions_initializers.single_swarm_uniform_positions_resetter(pso_instance, seed)
    elif isinstance(pso_instance, ClanPSO):
        pso_instance = positions_initializers.clan_swarms_uniform_positions_resetter(pso_instance, seed)


    current_gbest_value = pso_instance.gbest_value.copy()
    #* In cases where the goal point is not known
    #* (as would be the case in a neural network, for example),
    #* then the current error may be ignored,
    #* as long as it is also not included as a main loop termination criterion.
    current_error = np.linalg.norm(pso_instance.gbest_position - objective_function['goal_point'], ord=2)

    # Initialize the logging data structures with their first entry:
    # the initial state of the swarm.
    temp_indx = pd.Index(data=[0], dtype=int, name='iteration')

    df_positions_list = [pd.concat([swarm_positions_extractor(pso_instance)], keys=[0], names=['iteration'])]

    entry = gbest_position_extractor(pso_instance) ; entry.index = temp_indx
    df_gbests_list = [entry]

    df_gbest_values_list = [pd.DataFrame([current_gbest_value], index=temp_indx)]


    # Main loop of the algorithm
    i = 1
    while i <= maximum_iterations:
        pso_instance.step()
        if  pso_instance.gbest_value < current_gbest_value:
            current_gbest_value = pso_instance.gbest_value
            # current_error = np.linalg.norm(pso_instance.gbest_position - objective_function['goal_point'], ord=2)



        # Log START
        temp_indx = pd.Index([i], dtype=int, name='iteration')

        # All particles positions
        df_positions_list.append(pd.concat([swarm_positions_extractor(pso_instance)], keys=[i], names=['iteration']))
        # gbest positions
        entry = gbest_position_extractor(pso_instance) ; entry.index = temp_indx # Rename the index from default (incrementing integer) to iterationID.
        df_gbests_list.append(entry)
        # gbest values
        df_gbest_values_list.append(pd.DataFrame([current_gbest_value], index=temp_indx))

        # Log FINISH
        # Move to next iteration
        i += 1


    # Log the results of the experiment
    log_pso(data, ['positions', 'gbest_positions', 'gbest_values'], [df_positions_list, df_gbests_list, df_gbest_values_list])