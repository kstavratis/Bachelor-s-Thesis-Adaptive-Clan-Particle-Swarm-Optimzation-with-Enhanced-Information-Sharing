"""
Copyright (C) 2022  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

import pandas as pd
from concurrent.futures.process import ProcessPoolExecutor

from .experiment2 import experiment as experiment_script

from concurrent.futures.process import ProcessPoolExecutor


# TODO: Instead of writing each argument ("num_of_experiments", "objective_function_pointer", ...)
# explicitly over and over again, introduce a "kwargs" parameter, to pass into the executor.
# This will greatly reduce the number of import statements above!
def run(executor: ProcessPoolExecutor,
        num_of_experiments: int,
        *args,
        kwargs: dict,
        ):


    experiments_array = []


    objective_function_pointer = kwargs["objective_function_pointer"]; del kwargs['objective_function_pointer']
    swarm_size = kwargs["swarm_size"]; del kwargs["swarm_size"]
    is_clan = kwargs["is_clan"]; del kwargs["is_clan"]
    number_of_clans = kwargs["number_of_clans"]; del kwargs["number_of_clans"]


    for _ in range(num_of_experiments):
        function_task = executor.submit(experiment_script,
                                        objective_function_pointer,
                                        swarm_size,
                                        is_clan,
                                        number_of_clans,
                                        *args,
                                        kwargs = kwargs
                                        )

        experiments_array.append(function_task)

    #column_names = experiments_array[0].columns
    df = pd.DataFrame(columns=['Iteration', 'Centroid distance']) #! Lazy way of finding column names (i.e. hard-coded). Find a way to do it better

    result_dfs = [task.result() for task in experiments_array]
    result_dfs = [df.set_index('Iteration') for df in result_dfs]
    df = pd.concat(result_dfs, axis=1)
    # Renaming the dataframe's column names to "Experiment #1, Experiment #2, ..., Experiment #N", where N is the number of experiments conducted.
    df.columns = range(df.columns.size) # Reset column names from identical 'Centroid distance' to 0, ... N-1
    df.rename(lambda column: f'Experiment #{df.columns.get_loc(column) + 1}', axis='columns', inplace=True)

    return df