import scripts.experiments.experiment
import scripts.benchmark_functions as bench_f

from typing import Any, List, Tuple
from numpy import array, mean
from concurrent.futures.process import ProcessPoolExecutor

from classes.enums.enhanced_information_sharing.control_factor_types import ControlFactorTypes
from classes.enums.enhanced_information_sharing.global_local_coefficient_types import GlobalLocalCoefficientTypes
from classes.enums.wall_types import WallTypes


def run(num_of_experiments: int,
        objective_function_pointer: Any, spawn_boundaries: List[List[float]],
        objective_function_goal_point: array,
        maximum_iterations: int,
        swarm_size: int = 40, isClan: bool = False, clan_size: int = 4, c1: float = 2.0, c2: float = 2.0,
        adaptivePSO: bool = False,
        eis: Tuple[Tuple[GlobalLocalCoefficientTypes, float or None], Tuple[ControlFactorTypes, float or None]] =
        ((GlobalLocalCoefficientTypes.NONE, None), (ControlFactorTypes.NONE, None)),
        search_and_velocity_boundaries: List[List[float]] = None, wt: WallTypes = WallTypes.NONE
        ):

    experiments_precisions_per_experiment = []
    experiments_iterations_per_experiment = []
    experiments_average_iteration_cpu_times_per_experiment = []
    experiments_cpu_times_per_experiment = []

    executor = ProcessPoolExecutor()

    experiments_array = []

    for experiment in range(num_of_experiments):
        sphere_function_task = executor.submit(scripts.experiments.experiment.experiment,
                                               objective_function=objective_function_pointer,
                                               spawn_boundaries=spawn_boundaries,
                                               objective_function_goal_point=objective_function_goal_point,
                                               maximum_iterations=maximum_iterations,
                                               swarm_size=swarm_size, isClan=isClan, clan_size=clan_size,
                                               c1=c1, c2=c2,
                                               adaptivePSO=adaptivePSO,
                                               eis=eis,
                                               search_and_velocity_boundaries=search_and_velocity_boundaries, wt=wt
                                               )

        experiments_array.append(sphere_function_task)


    for task in experiments_array:
        precision, iterations, experiment_average_iteration_cpu_time, experiment_total_cpu_time = task.result()
        experiments_precisions_per_experiment.append(precision)
        experiments_iterations_per_experiment.append(iterations)
        experiments_average_iteration_cpu_times_per_experiment.append(experiment_average_iteration_cpu_time)
        experiments_cpu_times_per_experiment.append(experiment_total_cpu_time)


    mean_experiement_precision = mean(experiments_precisions_per_experiment)
    mean_experiment_iterations = mean(experiments_iterations_per_experiment)
    mean_experiments_average_iteration_cpu_times = mean(experiments_average_iteration_cpu_times_per_experiment)
    mean_experiment_cpu_time = mean(experiments_cpu_times_per_experiment)

    return mean_experiement_precision, mean_experiment_iterations, mean_experiments_average_iteration_cpu_times, \
           mean_experiment_cpu_time

