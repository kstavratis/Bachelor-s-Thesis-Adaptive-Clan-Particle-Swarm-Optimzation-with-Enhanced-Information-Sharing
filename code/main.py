"""
Code to conduct experiments with the Particle Swarm Optimization (PSO) and some of its variations (including ClanPSO and Adaptive PSO)
Code for my Bachelor's Thesis in the Informatics department of the Aristotle University of Thessaloniki (https://www.csd.auth.gr/en/)
    Copyright (C) 2021  Konstantinos Stavratis
    academic e-mail: kstavrat@csd.auth.gr
    alternative e-mail: kostauratis@gmail.com

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
from time import process_time

from scipy.constants import pi
from scipy.linalg import norm

from classes.PSOs.clanPSOswarm import ClanSwarm
from classes.enums.enhanced_information_sharing.control_factor_types import ControlFactorTypes
from classes.enums.enhanced_information_sharing.global_local_coefficient_types import GlobalLocalCoefficientTypes
from classes.enums.wall_types import WallTypes

import scripts.benchmark_functions as bench_f
import scripts.experiments.experiment
import scripts.experiments.experiments_data_creation

from numpy import array as vector, zeros, ones, seterr, mean, std, median, inf
import matplotlib.pyplot as plt
import pandas
from concurrent.futures.process import ProcessPoolExecutor


def main():
    seterr(all='raise')
    domain_dimensions = 10

    # Unimodal Functions search domain
    # sphere_function_search_domain = [[-inf, inf] for i in range(domain_dimensions)]
    sphere_function_search_domain = [[-10 ** 2, 10 ** 2] for i in range(domain_dimensions)]
    quadric_function_search_domain = [[-10 ** 2, 10 ** 2] for i in range(domain_dimensions)]
    schwefel222_function_search_domain = [[-10, 10] for i in range(domain_dimensions)]
    # rosenbrock_function_search_domain = [[-inf, inf] for i in range(domain_dimensions)]
    rosenbrock_function_search_domain = [[-10, 10] for i in range(domain_dimensions)]

    # Multimodal Functions search domain
    rastrigin_function_search_domain = [[-5.12, 5.12] for i in range(domain_dimensions)]
    ackley_function_search_domain = [[-32.768, 32.768] for i in range(domain_dimensions)]
    ackley_function_search_domain = [[-5, 5] for i in range(domain_dimensions)]
    salomon_function_search_domain = [[-10**2, 10**2] for i in range(domain_dimensions)]
    alpinen1_function_search_domain = [[0, 10] for i in range(domain_dimensions)]
    styblinski_tang_function_search_domain = [[-5, 5] for i in range(domain_dimensions)]



    # Unimodal Functions goal points
    sphere_function_goal_point = zeros(domain_dimensions)
    quadric_function_goal_point = zeros(domain_dimensions)
    schwefel222_function_goal_point = zeros(domain_dimensions)
    rosenbrock_function_goal_point = ones(domain_dimensions)


    # Multimodal Functions goal points
    rastrigin_function_goal_point = zeros(domain_dimensions)
    ackley_function_goal_point = zeros(domain_dimensions)
    salomon_function_goal_point = zeros(domain_dimensions)
    alpinen1_function_goal_point = zeros(domain_dimensions)


    fig, axis = plt.subplots()
    # axis.set_ylabel("Distance from target")
    # axis.plot(rastrigin_classic_divergences, marker=".", label="Classic PSO")
    # axis.plot(rastrigin_clan_divergences, marker=".", label="Clan PSO")
    # # axis.plot(rastrigin_classic_iterations,rastrigin_classic_errors, label="Classic PSO")
    # # axis.plot(rastrigin_clan_iterations, rastrigin_clan_errors, label="Clan PSO")
    # axis.legend()
    # plt.show()

    number_of_clans = 5
    particles_per_clan = 8
    simple_pso_particles = number_of_clans * particles_per_clan
    maximum_iterations = 5000
    experiments = 24
    executor = ProcessPoolExecutor()



    print("Schwefel P2.22 retest")
    print("---------------")

    # simple_AClanPSO = scripts.experiments.experiments_data_creation.run(
    #     executor=executor,
    #     num_of_experiments=experiments,
    #     objective_function_pointer=bench_f.salomon_function,
    #     spawn_boundaries=salomon_function_search_domain,
    #     objective_function_goal_point=salomon_function_goal_point,
    #     maximum_iterations=maximum_iterations,
    #     swarm_size=number_of_clans * particles_per_clan,
    #     isClan=True,
    #     number_of_clans=number_of_clans,
    #     adaptivePSO=True,
    #     search_and_velocity_boundaries=[[-100, 100], [-0.2 * 100, 0.2 * 100]],
    #     # wt=WallTypes.ELIMINATING
    # )
    #
    # eis_l2_k0p2c_AClanPSO = scripts.experiments.experiments_data_creation.run(
    #     executor=executor,
    #     num_of_experiments=experiments,
    #     objective_function_pointer=bench_f.salomon_function,
    #     spawn_boundaries=salomon_function_search_domain,
    #     objective_function_goal_point=salomon_function_goal_point,
    #     maximum_iterations=maximum_iterations,
    #     swarm_size=number_of_clans*particles_per_clan,
    #     isClan=True,
    #     number_of_clans=number_of_clans,
    #     adaptivePSO=True,
    #     eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.CONSTANT, 0.2)),
    #     search_and_velocity_boundaries=[[-100, 100], [-0.2 * 100, 0.2 * 100]],
    #     # wt=WallTypes.ELIMINATING
    # )
    #
    eis_l2_k1c_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=bench_f.schwefel222_function,
        spawn_boundaries=schwefel222_function_search_domain,
        objective_function_goal_point=schwefel222_function_goal_point,
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.CONSTANT, 1.0)),
        search_and_velocity_boundaries=[[-10, 10], [-0.2 * 10, 0.2 * 10]],
        # wt=WallTypes.ELIMINATING
    )

    # eis_l2_k2c_AClanPSO = scripts.experiments.experiments_data_creation.run(
    #     executor=executor,
    #     num_of_experiments=experiments,
    #     objective_function_pointer=bench_f.salomon_function,
    #     spawn_boundaries=salomon_function_search_domain,
    #     objective_function_goal_point=salomon_function_goal_point,
    #     maximum_iterations=maximum_iterations,
    #     swarm_size=number_of_clans * particles_per_clan,
    #     isClan=True,
    #     number_of_clans=number_of_clans,
    #     adaptivePSO=True,
    #     eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.CONSTANT, 2.0)),
    #     search_and_velocity_boundaries=[[-100, 100], [-0.2 * 100, 0.2 * 100]],
    #     # wt=WallTypes.ELIMINATING
    # )
    #
    # eis_l2_k0p2l_AClanPSO = scripts.experiments.experiments_data_creation.run(
    #     executor=executor,
    #     num_of_experiments=experiments,
    #     objective_function_pointer=bench_f.salomon_function,
    #     spawn_boundaries=salomon_function_search_domain,
    #     objective_function_goal_point=salomon_function_goal_point,
    #     maximum_iterations=maximum_iterations,
    #     swarm_size=number_of_clans * particles_per_clan,
    #     isClan=True,
    #     number_of_clans=number_of_clans,
    #     adaptivePSO=True,
    #     eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.LINEAR, 0.2)),
    #     search_and_velocity_boundaries=[[-100, 100], [-0.2 * 100, 0.2 * 100]],
    #     # wt=WallTypes.ELIMINATING
    # )

    eis_l2_k1l_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=bench_f.schwefel222_function,
        spawn_boundaries=schwefel222_function_search_domain,
        objective_function_goal_point=schwefel222_function_goal_point,
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.LINEAR, 1.0)),
        search_and_velocity_boundaries=[[-10, 10], [-0.2 * 10, 0.2 * 10]],
        # wt=WallTypes.ELIMINATING
    )

    # eis_l2_k0p2a_AClanPSO = scripts.experiments.experiments_data_creation.run(
    #     executor=executor,
    #     num_of_experiments=experiments,
    #     objective_function_pointer=bench_f.salomon_function,
    #     spawn_boundaries=salomon_function_search_domain,
    #     objective_function_goal_point=salomon_function_goal_point,
    #     maximum_iterations=maximum_iterations,
    #     swarm_size=number_of_clans * particles_per_clan,
    #     isClan=True,
    #     number_of_clans=number_of_clans,
    #     adaptivePSO=True,
    #     eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.ADAPTIVE, 0.2)),
    #     search_and_velocity_boundaries=[[-100, 100], [-0.2 * 100, 0.2 * 100]],
    #     # wt=WallTypes.ELIMINATING
    # )

    eis_l2_k1a_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=bench_f.schwefel222_function,
        spawn_boundaries=schwefel222_function_search_domain,
        objective_function_goal_point=schwefel222_function_goal_point,
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.ADAPTIVE, 1.0)),
        search_and_velocity_boundaries=[[-10, 10], [-0.2 * 10, 0.2 * 10]],
        # wt=WallTypes.ELIMINATING
    )



    sphere_collected_data = {
        # "Simple AClanPSO": pandas.Series([simple_AClanPSO["Precision mean"], simple_AClanPSO["Precision std"], simple_AClanPSO["Precision median"],
        #                                  simple_AClanPSO["Best precision"], simple_AClanPSO["Mean iterations per experiment"],
        #                                  simple_AClanPSO["Mean iteration CPU time"], simple_AClanPSO["Mean experiment CPU time"],
        #                                  maximum_iterations, experiments,
        #                                   number_of_clans, particles_per_clan],
        #                                 index=["Precision mean",
        #                                        "Precision std",
        #                                        "Precision median",
        #                                        "Best precision",
        #                                        "Mean iterations per experiment",
        #                                        "Mean iteration CPU time",
        #                                        "Mean experiment CPU time",
        #                                        "Maximum iterations per experiment",
        #                                        "Experiments",
        #                                        "No. clans",
        #                                        "Particles per clan"]),
        #
        # "EIS-L2-k0.2C AClanPSO": pandas.Series(
        #     [eis_l2_k0p2c_AClanPSO["Precision mean"], eis_l2_k0p2c_AClanPSO["Precision std"], eis_l2_k0p2c_AClanPSO["Precision median"],
        #      eis_l2_k0p2c_AClanPSO["Best precision"], eis_l2_k0p2c_AClanPSO["Mean iterations per experiment"],
        #      eis_l2_k0p2c_AClanPSO["Mean iteration CPU time"], eis_l2_k0p2c_AClanPSO["Mean experiment CPU time"],
        #      maximum_iterations, experiments,
        #      number_of_clans, particles_per_clan],
        #     index=["Precision mean",
        #            "Precision std",
        #            "Precision median",
        #            "Best precision",
        #            "Mean iterations per experiment",
        #            "Mean iteration CPU time",
        #            "Mean experiment CPU time",
        #            "Maximum iterations per experiment",
        #            "Experiments",
        #            "No. clans",
        #            "Particles per clan"]),
        #
        "EIS-L2-k1C AClanPSO": pandas.Series(
            [eis_l2_k1c_AClanPSO["Precision mean"], eis_l2_k1c_AClanPSO["Precision std"], eis_l2_k1c_AClanPSO["Precision median"],
             eis_l2_k1c_AClanPSO["Best precision"], eis_l2_k1c_AClanPSO["Mean iterations per experiment"],
             eis_l2_k1c_AClanPSO["Mean iteration CPU time"], eis_l2_k1c_AClanPSO["Mean experiment CPU time"],
             maximum_iterations, experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Maximum iterations per experiment",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]),
        #
        # "EIS-L2-k2C AClanPSO": pandas.Series(
        #     [eis_l2_k2c_AClanPSO["Precision mean"], eis_l2_k2c_AClanPSO["Precision std"], eis_l2_k2c_AClanPSO["Precision median"],
        #      eis_l2_k2c_AClanPSO["Best precision"], eis_l2_k2c_AClanPSO["Mean iterations per experiment"],
        #      eis_l2_k2c_AClanPSO["Mean iteration CPU time"], eis_l2_k2c_AClanPSO["Mean experiment CPU time"],
        #      maximum_iterations, experiments,
        #      number_of_clans, particles_per_clan],
        #     index=["Precision mean",
        #            "Precision std",
        #            "Precision median",
        #            "Best precision",
        #            "Mean iterations per experiment",
        #            "Mean iteration CPU time",
        #            "Mean experiment CPU time",
        #            "Maximum iterations per experiment",
        #            "Experiments",
        #            "No. clans",
        #            "Particles per clan"]),
        #
        # "EIS-L2-k0.2L AClanPSO": pandas.Series(
        #     [eis_l2_k0p2l_AClanPSO["Precision mean"], eis_l2_k0p2l_AClanPSO["Precision std"],
        #      eis_l2_k0p2l_AClanPSO["Precision median"],
        #      eis_l2_k0p2l_AClanPSO["Best precision"], eis_l2_k0p2l_AClanPSO["Mean iterations per experiment"],
        #      eis_l2_k0p2l_AClanPSO["Mean iteration CPU time"], eis_l2_k0p2l_AClanPSO["Mean experiment CPU time"],
        #      maximum_iterations, experiments,
        #      number_of_clans, particles_per_clan],
        #     index=["Precision mean",
        #            "Precision std",
        #            "Precision median",
        #            "Best precision",
        #            "Mean iterations per experiment",
        #            "Mean iteration CPU time",
        #            "Mean experiment CPU time",
        #            "Maximum iterations per experiment",
        #            "Experiments",
        #            "No. clans",
        #            "Particles per clan"]
        # ),

        "EIS-L2-k1L AClanPSO": pandas.Series(
            [eis_l2_k1l_AClanPSO["Precision mean"], eis_l2_k1l_AClanPSO["Precision std"],
             eis_l2_k1l_AClanPSO["Precision median"],
             eis_l2_k1l_AClanPSO["Best precision"], eis_l2_k1l_AClanPSO["Mean iterations per experiment"],
             eis_l2_k1l_AClanPSO["Mean iteration CPU time"], eis_l2_k1l_AClanPSO["Mean experiment CPU time"],
             maximum_iterations, experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Maximum iterations per experiment",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]
        ),

        # "EIS-L2-k0.2A AClanPSO": pandas.Series(
        #     [eis_l2_k0p2a_AClanPSO["Precision mean"], eis_l2_k0p2a_AClanPSO["Precision std"],
        #      eis_l2_k0p2a_AClanPSO["Precision median"],
        #      eis_l2_k0p2a_AClanPSO["Best precision"], eis_l2_k0p2a_AClanPSO["Mean iterations per experiment"],
        #      eis_l2_k0p2a_AClanPSO["Mean iteration CPU time"], eis_l2_k0p2a_AClanPSO["Mean experiment CPU time"],
        #      maximum_iterations, experiments,
        #      number_of_clans, particles_per_clan],
        #     index=["Precision mean",
        #            "Precision std",
        #            "Precision median",
        #            "Best precision",
        #            "Mean iterations per experiment",
        #            "Mean iteration CPU time",
        #            "Mean experiment CPU time",
        #            "Maximum iterations per experiment",
        #            "Experiments",
        #            "No. clans",
        #            "Particles per clan"]
        # ),

        "EIS-L2-k1A AClanPSO": pandas.Series(
            [eis_l2_k1a_AClanPSO["Precision mean"], eis_l2_k1a_AClanPSO["Precision std"],
             eis_l2_k1a_AClanPSO["Precision median"],
             eis_l2_k1a_AClanPSO["Best precision"], eis_l2_k1a_AClanPSO["Mean iterations per experiment"],
             eis_l2_k1a_AClanPSO["Mean iteration CPU time"], eis_l2_k1a_AClanPSO["Mean experiment CPU time"],
             maximum_iterations, experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Maximum iterations per experiment",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]
        )
    }

    pandas.DataFrame(sphere_collected_data).to_csv("schwefel corrected rule.csv")
    print("File for Salomon test function was created!")




if __name__ == "__main__":
    main()
