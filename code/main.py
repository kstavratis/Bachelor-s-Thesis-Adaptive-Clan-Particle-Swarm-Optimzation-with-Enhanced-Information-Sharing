import scripts.benchmark_functions as bench_f
import scripts.experiments.experiment
import scripts.experiments.experiments_data_creation
from numpy import array as vector, seterr
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager


def main():
    seterr(all='raise')
    domain_dimensions = 10


    rastrigin_function_search_domain = [[-5.12, 5.12] for i in range(domain_dimensions)]
    ackley_function_search_domain = [[-32.768, 32.768] for i in range(domain_dimensions)]
    ackley_function_search_domain = [[-5, 5] for i in range(domain_dimensions)]
    # sphere_function_search_domain = [[-inf, inf] for i in range(domain_dimensions)]
    sphere_function_search_domain = [[-10**3, 10**3] for i in range(domain_dimensions)]
    # The domain of Rosenbrock's function is x[i] ∈ [-∞, ∞] for all i
    # rosenbrock_function_search_domain = [[-inf, inf] for i in range(domain_dimensions)]
    rosenbrock_function_search_domain = [[-10**2, 10**2] for i in range(domain_dimensions)]
    styblinski_tang_function_search_domain = [[-5, 5] for i in range(domain_dimensions)]


    rastigin_function_goal_point = vector([0 for i in range(domain_dimensions)])
    ackley_function_goal_point = vector([0 for i in range(domain_dimensions)])
    sphere_function_goal_point = vector([0 for i in range(domain_dimensions)])
    rosenbrock_function_goal_point = vector([1 for i in range(domain_dimensions)])


    fig, axis = plt.subplots()
    # axis.set_ylabel("Distance from target")
    # axis.plot(rastrigin_classic_divergences, marker=".", label="Classic PSO")
    # axis.plot(rastrigin_clan_divergences, marker=".", label="Clan PSO")
    # # axis.plot(rastrigin_classic_iterations,rastrigin_classic_errors, label="Classic PSO")
    # # axis.plot(rastrigin_clan_iterations, rastrigin_clan_errors, label="Clan PSO")
    # axis.legend()
    # plt.show()

    # To achieve a balance between global and local exploration to speed up convergence to the true optimum,
    # an inertia weight whose value decreases linearly with the iteration number has been used.
    # The values of w_min = 0.4 and w_max = 0.9 are widely used.
    w_min, w_max = 0.4, 0.9

    clans = 5
    particles_per_clan = 8
    simple_pso_particles = clans * particles_per_clan
    maximum_iterations = 5000
    loop_stop_condition_limit = 5 * 10 ** (-15)
    experiments = 5

    print("Rastrigin Function")
    print("------------------")


    # rastrigin_adaptive_clan_precisions = []
    # rastrigin_adaptive_clan_cpu_average_iteration_times = []
    # rastrigin_adaptive_clan_experiment_cpu_times = []
    # rastrigin_adaptive_clan_iterations = []
    #
    # for experiment in range(experiments):
    #     print("Simple Adaptive Clan PSO: experiment #" + str(experiment + 1))
    #     loop_times = []
    #
    #     experiment_start = process_time()
    #
    #     rastrigin_adaptive_clan_swarm = ClanSwarm(fitness_function=rastrigin_function,
    #                                               spawn_boundaries=rastrigin_function_search_domain,
    #                                               number_of_clans=clans,
    #                                               swarm_size=particles_per_clan,
    #                                               maximum_iterations=maximum_iterations,
    #                                               adaptivePSO=True,
    #                                               search_and_velocity_boundaries=[[-5.12, 5.12], [-20 / 100 * 5.12, 20 / 100 * 5.12]],
    #                                               )
    #     iteration = 0
    #     loop_stop_condition_value = inf
    #
    #     while not(loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
    #         loop_start = process_time()
    #
    #         try:
    #             rastrigin_adaptive_clan_swarm.update_swarm()
    #         except FloatingPointError:
    #             iteration += 1
    #             loop_end = process_time()
    #             loop_times.append(loop_end - loop_start)
    #             break
    #
    #         loop_stop_condition_value = rastrigin_adaptive_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
    #         iteration += 1
    #
    #         loop_end = process_time()
    #         loop_times.append(loop_end - loop_start)
    #
    #     experiment_end = process_time()
    #     rastrigin_adaptive_clan_cpu_average_iteration_times.append(mean(loop_times))
    #     rastrigin_adaptive_clan_experiment_cpu_times.append(experiment_end - experiment_start)
    #     rastrigin_adaptive_clan_precisions.append(norm(
    #         rastrigin_adaptive_clan_swarm.find_population_global_best_position()
    #         - rastigin_function_goal_point
    #     ))
    #     rastrigin_adaptive_clan_iterations.append(iteration)
    #
    # rastrigin_adaptive_clan_mean_precision = mean(rastrigin_adaptive_clan_precisions)
    # rastrigin_adaptive_clan_precision_std = std(rastrigin_adaptive_clan_precisions)
    # rastrigin_adaptive_clan_precision_median = median(rastrigin_adaptive_clan_precisions)
    # rastrigin_adaptive_clan_precision_best = min(rastrigin_adaptive_clan_precisions)
    # rastrigin_adaptive_clan_average_iteration_cpu_time = mean(rastrigin_adaptive_clan_cpu_average_iteration_times)
    # rastrigin_adaptive_clan_mean_experiment_cpu_time = mean(rastrigin_adaptive_clan_experiment_cpu_times)
    # rastrigin_adaptive_clan_mean_iterations = mean(rastrigin_adaptive_clan_iterations)
    #
    #
    #
    #
    # [print() for i in range(3)]




    # rastrigin_eis_k0point2_adaptive_clan_precisions = []
    # rastrigin_eis_k0point2_adaptive_clan_cpu_average_iteration_times = []
    # rastrigin_eis_k0point2_adaptive_clan_experiment_cpu_times = []
    # rastrigin_eis_k0point2_adaptive_clan_iterations = []
    #
    # for experiment in range(experiments):
    #     print("EIS-L-k0.2A AClanPSO: experiment #" + str(experiment + 1))
    #
    #     loop_times = []
    #
    #     experiment_start = process_time()
    #
    #     rastrigin_eis_adaptive_clan_swarm = ClanSwarm(fitness_function=bench_f.rastrigin_function,
    #                                                   spawn_boundaries=rastrigin_function_search_domain,
    #                                                   number_of_clans=clans,
    #                                                   swarm_size=particles_per_clan,
    #                                                   maximum_iterations=maximum_iterations,
    #                                                   adaptivePSO=True,
    #                                                   search_and_velocity_boundaries=[[-5.12, 5.12], [-20 / 100 * 5.12,
    #                                                                                                   20 / 100 * 5.12]],
    #                                                   eis=((GlobalLocalCoefficientTypes.LINEAR, None), (ControlFactorTypes.ADAPTIVE, 0.2))
    #                                                   )
    #     iteration = 0
    #     loop_stop_condition_value = inf
    #
    #     while not(loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
    #         loop_start = process_time()
    #
    #         try:
    #             rastrigin_eis_adaptive_clan_swarm.update_swarm()
    #         except FloatingPointError:
    #             iteration += 1
    #             loop_end = process_time()
    #             loop_times.append(loop_end - loop_start)
    #             break
    #
    #         loop_stop_condition_value = rastrigin_eis_adaptive_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
    #         iteration += 1
    #
    #         loop_end = process_time()
    #         loop_times.append(loop_end - loop_start)
    #
    #     experiment_end = process_time()
    #     rastrigin_eis_k0point2_adaptive_clan_cpu_average_iteration_times.append(mean(loop_times))
    #     rastrigin_eis_k0point2_adaptive_clan_experiment_cpu_times.append(experiment_end - experiment_start)
    #     rastrigin_eis_k0point2_adaptive_clan_precisions.append(norm(
    #         rastrigin_eis_adaptive_clan_swarm.find_population_global_best_position()
    #         - rastigin_function_goal_point
    #     ))
    #     rastrigin_eis_k0point2_adaptive_clan_iterations.append(iteration)
    #
    # rastrigin_eis_k0point2_adaptive_clan_mean_precision = mean(rastrigin_eis_k0point2_adaptive_clan_precisions)
    # rastrigin_eis_k0point2_adaptive_clan_precision_std = std(rastrigin_eis_k0point2_adaptive_clan_precisions)
    # rastrigin_eis_k0point2_adaptive_clan_precision_median = median(rastrigin_eis_k0point2_adaptive_clan_precisions)
    # rastrigin_eis_k0point2_adaptive_clan_precision_best = min(rastrigin_eis_k0point2_adaptive_clan_precisions)
    # rastrigin_eis_k0point2_adaptive_clan_average_iteration_cpu_time = mean(rastrigin_eis_k0point2_adaptive_clan_cpu_average_iteration_times)
    # rastrigin_eis_k0point2_adaptive_clan_mean_experiment_cpu_time = mean(rastrigin_eis_k0point2_adaptive_clan_experiment_cpu_times)
    # rastrigin_eis_k0point2_adaptive_clan_mean_iterations = mean(rastrigin_eis_k0point2_adaptive_clan_iterations)
    #
    #
    # [print() for i in range(2)]
    #
    #
    # rastrigin_eis_k2_adaptive_clan_precisions = []
    # rastrigin_eis_k2_adaptive_clan_cpu_average_iteration_times = []
    # rastrigin_eis_k2_adaptive_clan_experiment_cpu_times = []
    # rastrigin_eis_k2_adaptive_clan_iterations = []
    #
    # for experiment in range(experiments):
    #     print("EIS-L-k2A AClanPSO: experiment #" + str(experiment + 1))
    #
    #     loop_times = []
    #
    #     experiment_start = process_time()
    #
    #     rastrigin_eis_adaptive_clan_swarm = ClanSwarm(fitness_function=bench_f.rastrigin_function,
    #                                                   spawn_boundaries=rastrigin_function_search_domain,
    #                                                   number_of_clans=clans,
    #                                                   swarm_size=particles_per_clan,
    #                                                   maximum_iterations=maximum_iterations,
    #                                                   adaptivePSO=True,
    #                                                   search_and_velocity_boundaries=[[-5.12, 5.12], [-20 / 100 * 5.12,
    #                                                                                                   20 / 100 * 5.12]],
    #                                                   eis=((GlobalLocalCoefficientTypes.LINEAR, None),(ControlFactorTypes.ADAPTIVE, 2.0))
    #                                                   )
    #     iteration = 0
    #     loop_stop_condition_value = inf
    #
    #     while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
    #         loop_start = process_time()
    #
    #         try:
    #             rastrigin_eis_adaptive_clan_swarm.update_swarm()
    #         except FloatingPointError:
    #             iteration += 1
    #             loop_end = process_time()
    #             loop_times.append(loop_end - loop_start)
    #             break
    #
    #         loop_stop_condition_value = rastrigin_eis_adaptive_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
    #         iteration += 1
    #
    #         loop_end = process_time()
    #         loop_times.append(loop_end - loop_start)
    #
    #     experiment_end = process_time()
    #     rastrigin_eis_k2_adaptive_clan_cpu_average_iteration_times.append(mean(loop_times))
    #     rastrigin_eis_k2_adaptive_clan_experiment_cpu_times.append(experiment_end - experiment_start)
    #     rastrigin_eis_k2_adaptive_clan_precisions.append(norm(
    #         rastrigin_eis_adaptive_clan_swarm.find_population_global_best_position()
    #         - rastigin_function_goal_point
    #     ))
    #     rastrigin_eis_k2_adaptive_clan_iterations.append(iteration)
    #
    # rastrigin_eis_k2_adaptive_clan_mean_precision = mean(rastrigin_eis_k2_adaptive_clan_precisions)
    # rastrigin_eis_k2_adaptive_clan_precision_std = std(rastrigin_eis_k2_adaptive_clan_precisions)
    # rastrigin_eis_k2_adaptive_clan_precision_median = median(rastrigin_eis_k2_adaptive_clan_precisions)
    # rastrigin_eis_k2_adaptive_clan_precision_best = min(rastrigin_eis_k2_adaptive_clan_precisions)
    # rastrigin_eis_k2_adaptive_clan_average_iteration_cpu_time = mean(
    #     rastrigin_eis_k2_adaptive_clan_cpu_average_iteration_times)
    # rastrigin_eis_k2_adaptive_clan_mean_experiment_cpu_time = mean(
    #     rastrigin_eis_k2_adaptive_clan_experiment_cpu_times)
    # rastrigin_eis_k2_adaptive_clan_mean_iterations = mean(rastrigin_eis_k2_adaptive_clan_iterations)
    #
    # rastrigin_collected_data = {
    #     # "Simple AClanPSO": pandas.Series([rastrigin_adaptive_clan_mean_precision,
    #     #                                   rastrigin_adaptive_clan_precision_std,
    #     #                                   rastrigin_adaptive_clan_precision_median,
    #     #                                   rastrigin_adaptive_clan_precision_best,
    #     #                                   rastrigin_adaptive_clan_mean_experiment_cpu_time,
    #     #                                   rastrigin_adaptive_clan_average_iteration_cpu_time,
    #     #                                   experiments,
    #     #                                   rastrigin_adaptive_clan_mean_iterations,
    #     #                                   maximum_iterations],
    #     #                                  index=["Precision mean",
    #     #                                         "Precision std",
    #     #                                         "Precision median",
    #     #                                         "Best precision",
    #     #                                         "Mean experiment CPU time",
    #     #                                         "Mean iteration CPU time",
    #     #                                         "Experiments",
    #     #                                         "Mean iterations per experiment",
    #     #                                         "Maximum iterations"]),
    #
    #     "EIS-L-k0.2A AClanPSO": pandas.Series([rastrigin_eis_k0point2_adaptive_clan_mean_precision,
    #                                                   rastrigin_eis_k0point2_adaptive_clan_precision_std,
    #                                                   rastrigin_eis_k0point2_adaptive_clan_precision_median,
    #                                                   rastrigin_eis_k0point2_adaptive_clan_precision_best,
    #                                                   rastrigin_eis_k0point2_adaptive_clan_mean_experiment_cpu_time,
    #                                                   rastrigin_eis_k0point2_adaptive_clan_average_iteration_cpu_time,
    #                                                   experiments,
    #                                                   rastrigin_eis_k0point2_adaptive_clan_mean_iterations,
    #                                                   maximum_iterations],
    #                                                  index=["Precision mean",
    #                                                         "Precision std",
    #                                                         "Precision median",
    #                                                         "Best precision",
    #                                                         "Mean experiment CPU time",
    #                                                         "Mean iteration CPU time",
    #                                                         "Experiments",
    #                                                         "Mean iterations per experiment",
    #                                                         "Maximum iterations per experiment"]) ,
    #
    #     "EIS-L-k2A AClanPSO": pandas.Series([rastrigin_eis_k2_adaptive_clan_mean_precision,
    #                                            rastrigin_eis_k2_adaptive_clan_precision_std,
    #                                            rastrigin_eis_k2_adaptive_clan_precision_median,
    #                                            rastrigin_eis_k2_adaptive_clan_precision_best,
    #                                            rastrigin_eis_k2_adaptive_clan_mean_experiment_cpu_time,
    #                                            rastrigin_eis_k2_adaptive_clan_average_iteration_cpu_time,
    #                                            experiments,
    #                                            rastrigin_eis_k2_adaptive_clan_mean_iterations,
    #                                            maximum_iterations],
    #                                           index=["Precision mean",
    #                                                  "Precision std",
    #                                                  "Precision median",
    #                                                  "Best precision",
    #                                                  "Mean experiment CPU time",
    #                                                  "Mean iteration CPU time",
    #                                                  "Experiments",
    #                                                  "Mean iterations per experiment",
    #                                                  "Maximum iterations per experiment"]),
    #
    #
    # }
    #
    # [print() for i in range(3)]
    # print("Creating file for Rastrigin test function comparison...")
    # pandas.DataFrame(rastrigin_collected_data).to_csv(r'rastrigin function.csv')
    # print("File for Rastrigin test function created!")
    #
    #
    #
    #
    #
    # # [print() for i in range(6)]
    # #
    # #
    # # print("Sphere Function")
    # # print("---------------")
    # #
    # # sphere_adaptive_clan_precisions = []
    # # sphere_adaptive_clan_cpu_average_iteration_times = []
    # # sphere_adaptive_clan_experiment_cpu_times = []
    # # sphere_adaptive_clan_iterations = []
    # #
    # # for experiment in range(experiments):
    # #     print("Simple Adaptive Clan PSO: experiment #" + str(experiment + 1))
    # #     loop_times = []
    # #
    # #     experiment_start = process_time()
    # #
    # #     sphere_adaptive_clan_swarm = ClanSwarm(fitness_function=sphere_function,
    # #                                            spawn_boundaries=sphere_function_search_domain,
    # #                                            maximum_iterations=maximum_iterations,
    # #                                            adaptivePSO=True,
    # #                                            # search_and_velocity_boundaries=[[-5.12, 5.12],
    # #                                            #                                 [-20 / 100 * 5.12, 20 / 100 * 5.12]],
    # #                                            eis=((GlobalLocalCoefficientTypes.NONE, None),(ControlFactorTypes.NONE, None))
    # #                                            )
    # #     iteration = 0
    # #     loop_stop_condition_value = inf
    # #
    # #     while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
    # #         loop_start = process_time()
    # #
    # #         try:
    # #             sphere_adaptive_clan_swarm.update_swarm()
    # #         except FloatingPointError:
    # #             iteration += 1
    # #             loop_end = process_time()
    # #             loop_times.append(loop_end - loop_start)
    # #             break
    # #
    # #         loop_stop_condition_value = sphere_adaptive_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
    # #         iteration += 1
    # #
    # #         loop_end = process_time()
    # #         loop_times.append(loop_end - loop_start)
    # #
    # #     experiment_end = process_time()
    # #     sphere_adaptive_clan_cpu_average_iteration_times.append(mean(loop_times))
    # #     sphere_adaptive_clan_experiment_cpu_times.append(experiment_end - experiment_start)
    # #     sphere_adaptive_clan_precisions.append(norm(
    # #         sphere_adaptive_clan_swarm.find_population_global_best_position()
    # #         - sphere_function_goal_point
    # #     ))
    # #     sphere_adaptive_clan_iterations.append(iteration)
    # #
    # # sphere_adaptive_clan_mean_precision = mean(sphere_adaptive_clan_precisions)
    # # sphere_adaptive_clan_precision_std = std(sphere_adaptive_clan_precisions)
    # # sphere_adaptive_clan_precision_median = median(sphere_adaptive_clan_precisions)
    # # sphere_adaptive_clan_precision_best = min(sphere_adaptive_clan_precisions)
    # # sphere_adaptive_clan_average_iteration_cpu_time = mean(sphere_adaptive_clan_cpu_average_iteration_times)
    # # sphere_adaptive_clan_mean_experiment_cpu_time = mean(sphere_adaptive_clan_experiment_cpu_times)
    # # sphere_adaptive_clan_mean_iterations = mean(sphere_adaptive_clan_iterations)
    # #
    # #
    # #
    # #
    # # [print() for i in range(2)]
    #
    #
    #
    #
    # sphere_eis_k0point2_adaptive_clan_precisions = []
    # sphere_eis_k0point2_adaptive_clan_cpu_average_iteration_times = []
    # sphere_eis_k0point2_adaptive_clan_experiment_cpu_times = []
    # sphere_eis_k0point2_adaptive_clan_iterations = []
    #
    # for experiment in range(experiments):
    #     print("EIS-L-k0.2A AClanPSO: experiment #" + str(experiment + 1))
    #
    #     loop_times = []
    #
    #     experiment_start = process_time()
    #
    #     sphere_eis_adaptive_clan_swarm = ClanSwarm(fitness_function=bench_f.sphere_function,
    #                                                spawn_boundaries=rastrigin_function_search_domain,
    #                                                number_of_clans=clans,
    #                                                swarm_size=particles_per_clan,
    #                                                maximum_iterations=maximum_iterations,
    #                                                adaptivePSO=True,
    #                                                # search_and_velocity_boundaries=[[-5.12, 5.12], [-20 / 100 * 5.12,
    #                                                #                                                 20 / 100 * 5.12]],
    #                                                eis=((GlobalLocalCoefficientTypes.LINEAR, None),(ControlFactorTypes.ADAPTIVE, 0.2))
    #                                                )
    #
    #     iteration = 0
    #     loop_stop_condition_value = inf
    #
    #     while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
    #         loop_start = process_time()
    #
    #         try:
    #             sphere_eis_adaptive_clan_swarm.update_swarm()
    #         except FloatingPointError:
    #             iteration += 1
    #             loop_end = process_time()
    #             loop_times.append(loop_end - loop_start)
    #             break
    #
    #         loop_stop_condition_value = sphere_eis_adaptive_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
    #         iteration += 1
    #
    #         loop_end = process_time()
    #         loop_times.append(loop_end - loop_start)
    #
    #     experiment_end = process_time()
    #     sphere_eis_k0point2_adaptive_clan_cpu_average_iteration_times.append(mean(loop_times))
    #     sphere_eis_k0point2_adaptive_clan_experiment_cpu_times.append(experiment_end - experiment_start)
    #     sphere_eis_k0point2_adaptive_clan_precisions.append(norm(
    #         sphere_eis_adaptive_clan_swarm.find_population_global_best_position()
    #         - sphere_function_goal_point
    #     ))
    #     sphere_eis_k0point2_adaptive_clan_iterations.append(iteration)
    #
    # sphere_eis_k0point2_adaptive_clan_mean_precision = mean(sphere_eis_k0point2_adaptive_clan_precisions)
    # sphere_eis_k0point2_adaptive_clan_precision_std = std(sphere_eis_k0point2_adaptive_clan_precisions)
    # sphere_eis_k0point2_adaptive_clan_precision_median = median(sphere_eis_k0point2_adaptive_clan_precisions)
    # sphere_eis_k0point2_adaptive_clan_precision_best = min(sphere_eis_k0point2_adaptive_clan_precisions)
    # sphere_eis_k0point2_adaptive_clan_average_iteration_cpu_time = mean(
    #     sphere_eis_k0point2_adaptive_clan_cpu_average_iteration_times)
    # sphere_eis_k0point2_adaptive_clan_mean_experiment_cpu_time = mean(sphere_eis_k0point2_adaptive_clan_experiment_cpu_times)
    # sphere_eis_k0point2_adaptive_clan_mean_iterations = mean(sphere_eis_k0point2_adaptive_clan_iterations)
    #
    #
    # [print() for i in range(2)]
    #
    #
    # sphere_eis_k1_adaptive_clan_precisions = []
    # sphere_eis_k1_adaptive_clan_cpu_average_iteration_times = []
    # sphere_eis_k1_adaptive_clan_experiment_cpu_times = []
    # sphere_eis_k1_adaptive_clan_iterations = []
    #
    # for experiment in range(experiments):
    #     print("EIS-L-k2A AClanPSO: experiment #" + str(experiment + 1))
    #
    #     loop_times = []
    #
    #     experiment_start = process_time()
    #
    #     sphere_eis_adaptive_clan_swarm = ClanSwarm(fitness_function=bench_f.sphere_function,
    #                                                spawn_boundaries=rastrigin_function_search_domain,
    #                                                number_of_clans=clans,
    #                                                swarm_size=particles_per_clan,
    #                                                maximum_iterations=maximum_iterations,
    #                                                adaptivePSO=True,
    #                                                # search_and_velocity_boundaries=[[-5.12, 5.12], [-20 / 100 * 5.12,
    #                                                #                                                 20 / 100 * 5.12]],
    #                                                eis=((GlobalLocalCoefficientTypes.LINEAR, None),(ControlFactorTypes.ADAPTIVE, 2.0))
    #                                                )
    #     iteration = 0
    #     loop_stop_condition_value = inf
    #
    #     while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
    #         loop_start = process_time()
    #
    #         try:
    #             sphere_eis_adaptive_clan_swarm.update_swarm()
    #         except FloatingPointError:
    #             iteration += 1
    #             loop_end = process_time()
    #             loop_times.append(loop_end - loop_start)
    #             break
    #
    #         loop_stop_condition_value = sphere_eis_adaptive_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
    #         iteration += 1
    #
    #         loop_end = process_time()
    #         loop_times.append(loop_end - loop_start)
    #
    #     experiment_end = process_time()
    #     sphere_eis_k1_adaptive_clan_cpu_average_iteration_times.append(mean(loop_times))
    #     sphere_eis_k1_adaptive_clan_experiment_cpu_times.append(experiment_end - experiment_start)
    #     sphere_eis_k1_adaptive_clan_precisions.append(norm(
    #         sphere_eis_adaptive_clan_swarm.find_population_global_best_position()
    #         - sphere_function_goal_point
    #     ))
    #     sphere_eis_k1_adaptive_clan_iterations.append(iteration)
    #
    # sphere_eis_k1_adaptive_clan_mean_precision = mean(sphere_eis_k1_adaptive_clan_precisions)
    # sphere_eis_k1_adaptive_clan_precision_std = std(sphere_eis_k1_adaptive_clan_precisions)
    # sphere_eis_k1_adaptive_clan_precision_median = median(sphere_eis_k1_adaptive_clan_precisions)
    # sphere_eis_k1_adaptive_clan_precision_best = min(sphere_eis_k1_adaptive_clan_precisions)
    # sphere_eis_k1_adaptive_clan_average_iteration_cpu_time = mean(
    #     sphere_eis_k1_adaptive_clan_cpu_average_iteration_times)
    # sphere_eis_k1_adaptive_clan_mean_experiment_cpu_time = mean(
    #     sphere_eis_k1_adaptive_clan_experiment_cpu_times)
    # sphere_eis_k1_adaptive_clan_mean_iterations = mean(sphere_eis_k1_adaptive_clan_iterations)
    #
    # sphere_collected_data = {
    #     # "Simple AClanPSO": pandas.Series([sphere_adaptive_clan_mean_precision,
    #     #                                   sphere_adaptive_clan_precision_std,
    #     #                                   sphere_adaptive_clan_precision_median,
    #     #                                   sphere_adaptive_clan_precision_best,
    #     #                                   sphere_adaptive_clan_mean_experiment_cpu_time,
    #     #                                   sphere_adaptive_clan_average_iteration_cpu_time,
    #     #                                   experiments,
    #     #                                   sphere_adaptive_clan_mean_iterations,
    #     #                                   maximum_iterations],
    #     #                                  index=["Precision mean",
    #     #                                         "Precision std",
    #     #                                         "Precision median",
    #     #                                         "Best precision",
    #     #                                         "Mean experiment CPU time",
    #     #                                         "Mean iteration CPU time",
    #     #                                         "Experiments",
    #     #                                         "Mean iterations per experiment",
    #     #                                         "Maximum iterations per experiment"]),
    #
    #     "EIS-L-k0.2A AClanPSO": pandas.Series([sphere_eis_k0point2_adaptive_clan_mean_precision,
    #                                    sphere_eis_k0point2_adaptive_clan_precision_std,
    #                                    sphere_eis_k0point2_adaptive_clan_precision_median,
    #                                    sphere_eis_k0point2_adaptive_clan_precision_best,
    #                                    sphere_eis_k0point2_adaptive_clan_mean_experiment_cpu_time,
    #                                    sphere_eis_k0point2_adaptive_clan_average_iteration_cpu_time,
    #                                                   experiments,
    #                                    sphere_eis_k0point2_adaptive_clan_mean_iterations,
    #                                                   maximum_iterations],
    #                                   index=["Precision mean",
    #                                          "Precision std",
    #                                          "Precision median",
    #                                          "Best precision",
    #                                          "Mean experiment CPU time",
    #                                          "Mean iteration CPU time",
    #                                          "Experiments",
    #                                          "Mean iterations per experiment",
    #                                          "Maximum iterations per experiment"]),
    #
    #     "EIS-L-k2A AClanPSO": pandas.Series([sphere_eis_k1_adaptive_clan_mean_precision,
    #                                          sphere_eis_k1_adaptive_clan_precision_std,
    #                                          sphere_eis_k1_adaptive_clan_precision_median,
    #                                          sphere_eis_k1_adaptive_clan_precision_best,
    #                                          sphere_eis_k1_adaptive_clan_mean_experiment_cpu_time,
    #                                          sphere_eis_k1_adaptive_clan_average_iteration_cpu_time,
    #                                          experiments,
    #                                          sphere_eis_k1_adaptive_clan_mean_iterations,
    #                                          maximum_iterations],
    #                                         index=["Precision mean",
    #                                                "Precision std",
    #                                                "Precision median",
    #                                                "Best precision",
    #                                                "Mean experiment CPU time",
    #                                                "Mean iteration CPU time",
    #                                                "Experiments",
    #                                                "Mean iterations per experiment",
    #                                                "Maximum iterations per experiment"])
    # }
    #
    # [print() for i in range(3)]
    # print("Creating file for Sphere test function comparison...")
    # pandas.DataFrame(sphere_collected_data).to_csv(r'sphere function.csv')
    # print("File for Sphere test function created!")

    # sphere_precision = []
    # sphere_iterations = []
    # sphere_average_iteration_cpu_times = []
    # sphere_experiment_cpu_times = []
    #
    # executor = ProcessPoolExecutor()
    #
    # experiments_array = []
    #
    # for experiment in range(experiments):
    #     sphere_function_task = executor.submit(scripts.experiments.experiment.experiment,
    #                                            objective_function=bench_f.sphere_function,
    #                                            function_boundaries=sphere_function_search_domain,
    #                                            objective_function_goal_point=sphere_function_goal_point,
    #                                            maximum_iterations=maximum_iterations, isClan=True, adaptivePSO=True)
    #
    #     experiments_array.append(sphere_function_task)
    #
    #
    # for task in experiments_array:
    #     precision, iterations, experiment_average_iteration_cpu_time, experiment_total_cpu_time = task.result()
    #     sphere_precision.append(precision)
    #     sphere_iterations.append(iterations)
    #     sphere_average_iteration_cpu_times.append(experiment_average_iteration_cpu_time)
    #     sphere_experiment_cpu_times.append(experiment_total_cpu_time)
    #
    # print(sphere_precision)
    # print(sphere_iterations)
    # print(sphere_average_iteration_cpu_times)
    # print(sphere_experiment_cpu_times)

    test = scripts.experiments.experiments_data_creation.run(
        num_of_experiments=10,
        objective_function_pointer=bench_f.sphere_function,
        spawn_boundaries=sphere_function_search_domain,
        objective_function_goal_point=sphere_function_goal_point,
        maximum_iterations=maximum_iterations,
        isClan=True,
        adaptivePSO=True
    )

    print("mean experiment precision, mean experiment iterations, mean experiment average iteration cpu time, mean experiement cpu time =")
    print(test)


if __name__ == "__main__":
    main()