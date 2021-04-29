"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from typing import Any, List, Tuple
from numpy import array, mean, std, median
from concurrent.futures.process import ProcessPoolExecutor

from classes.enums.enhanced_information_sharing.control_factor_types import ControlFactorTypes
from classes.enums.enhanced_information_sharing.global_local_coefficient_types import GlobalLocalCoefficientTypes
from classes.enums.wall_types import WallTypes

import scripts.experiments.experiment


def run(executor: ProcessPoolExecutor,
        num_of_experiments: int,
        objective_function_pointer: Any, spawn_boundaries: List[List[float]],
        objective_function_goal_point: array,
        maximum_iterations: int,
        swarm_size: int = 40, isClan: bool = False, number_of_clans: int = 4, c1: float = 2.0, c2: float = 2.0,
        adaptivePSO: bool = False,
        eis: Tuple[Tuple[GlobalLocalCoefficientTypes, float or None], Tuple[ControlFactorTypes, float or None]] =
        ((GlobalLocalCoefficientTypes.NONE, None), (ControlFactorTypes.NONE, None)),
        search_and_velocity_boundaries: List[List[float]] = None, wt: WallTypes = WallTypes.NONE
        ):
    experiments_precisions_per_experiment = []
    experiments_iterations_per_experiment = []
    experiments_average_iteration_cpu_times_per_experiment = []
    experiments_cpu_times_per_experiment = []

    experiments_array = []

    for experiment in range(num_of_experiments):
        function_task = executor.submit(scripts.experiments.experiment.experiment,
                                        objective_function=objective_function_pointer,
                                        spawn_boundaries=spawn_boundaries,
                                        objective_function_goal_point=objective_function_goal_point,
                                        maximum_iterations=maximum_iterations,
                                        swarm_size=swarm_size, isClan=isClan, number_of_clans=number_of_clans,
                                        c1=c1, c2=c2,
                                        adaptivePSO=adaptivePSO,
                                        eis=eis,
                                        search_and_velocity_boundaries=search_and_velocity_boundaries, wt=wt
                                        )

        experiments_array.append(function_task)

    for task in experiments_array:
        precision, iterations, experiment_average_iteration_cpu_time, experiment_total_cpu_time = task.result()
        experiments_precisions_per_experiment.append(precision)
        experiments_iterations_per_experiment.append(iterations)
        experiments_average_iteration_cpu_times_per_experiment.append(experiment_average_iteration_cpu_time)
        experiments_cpu_times_per_experiment.append(experiment_total_cpu_time)

    mean_experiment_precision = mean(experiments_precisions_per_experiment)
    std_experiment_precision = std(experiments_precisions_per_experiment)
    median_experiment_precision = median(experiments_precisions_per_experiment)
    best_experiment_precision = min(experiments_precisions_per_experiment)
    mean_experiment_iterations = mean(experiments_iterations_per_experiment)
    mean_experiments_average_iteration_cpu_times = mean(experiments_average_iteration_cpu_times_per_experiment)
    mean_experiment_cpu_time = mean(experiments_cpu_times_per_experiment)

    return {"Precision mean": mean_experiment_precision,
            "Precision std": std_experiment_precision,
            "Precision median": median_experiment_precision,
            "Best precision": best_experiment_precision,
            "Mean iterations per experiment": mean_experiment_iterations,
            "Mean iteration CPU time": mean_experiments_average_iteration_cpu_times,
            "Mean experiment CPU time": mean_experiment_cpu_time}
