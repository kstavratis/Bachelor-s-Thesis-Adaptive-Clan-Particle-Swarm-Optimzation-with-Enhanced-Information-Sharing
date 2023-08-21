"""
Copyright (C) 2023  Konstantinos Stavratis
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
from src.scripts.io_handling.data_generation.extractor import swarm_positions_extractor, gbest_extractor
from src.scripts.io_handling.data_generation.logger import log_pso
import src.scripts.benchmark_functions as bf 

from ..io_handling.data_generation.logger import log_pso

import numpy as np
import pandas as pd


def run(data):
    # Hyperparameter declarations
    maximum_iterations = None
    objective_function = None

    # Variable declarations
    current_error, numerical_precision_termination_criterion = None , 1e-14

    # Variable declarations
    pso_instance = None


    maximum_iterations = data['max_iterations']
    objective_function = getattr(bf, data['objective_function'] + '_function')
    bf.domain_dimensions = data['nr_dimensions']


    pso_instance = handle_config_file_data(data)


    current_gbest_value = pso_instance.gbest_value.copy()
    current_error = np.linalg.norm(pso_instance.gbest_position - objective_function['goal_point'], ord=2)

    # Initialize the logging data structures with their first entry:
    # the initial state of the swarm.
    temp_indx = pd.Index(data=[0], dtype=int, name='iteration')

    df_positions_list = [pd.concat([swarm_positions_extractor(pso_instance)], keys=[0], names=['iteration'])]

    entry = gbest_extractor(pso_instance) ; entry.index = temp_indx
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
        entry = gbest_extractor(pso_instance) ; entry.index = temp_indx # Rename the index from default (incrementing integer) to iterationID.
        df_gbests_list.append(entry)
        # gbest values
        df_gbest_values_list.append(pd.DataFrame([current_gbest_value], index=temp_indx))

        # Log FINISH
        # Move to next iteration
        i += 1


    # Log the results of the experiment
    log_pso(data, ['positions', 'gbest_positions', 'gbest_values'], [df_positions_list, df_gbests_list, df_gbest_values_list])