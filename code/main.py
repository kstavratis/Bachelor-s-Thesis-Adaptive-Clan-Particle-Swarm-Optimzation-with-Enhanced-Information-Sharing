"""
Code to conduct experiments with the Particle Swarm Optimization (PSO) and some of its variations (including ClanPSO and Adaptive PSO)
Code for my Bachelor's Thesis in the Informatics department of the Aristotle University of Thessaloniki (https://www.csd.auth.gr/en/)
    Copyright (C) 2022  Konstantinos Stavratis
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

from classes.PSOs.clanPSOswarm import ClanSwarm
from classes.enums.enhanced_information_sharing.control_factor_types import ControlFactorTypes
from classes.enums.enhanced_information_sharing.global_local_coefficient_types import GlobalLocalCoefficientTypes
from classes.enums.wall_types import WallTypes

import scripts.benchmark_functions as bench_f
import scripts.experiments.experiment
import scripts.experiments.experiments_data_creation

from numpy import array as vector, zeros, ones, seterr, mean, std, median, inf
import pandas
from concurrent.futures.process import ProcessPoolExecutor


def main():
    seterr(all='raise')


    benchmark_function = bench_f.ackley_function


    number_of_clans = 6
    particles_per_clan = 5
    simple_pso_particles = number_of_clans * particles_per_clan
    maximum_iterations = 5000
    experiments = 256
    executor = ProcessPoolExecutor()

    print(f'Alpine N.1 function: {number_of_clans} clans of {particles_per_clan} particles each')
    print("---------------")

    simple_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=benchmark_function['formula'],
        spawn_boundaries=benchmark_function['search_domain'],
        objective_function_goal_point=benchmark_function['goal_point'],
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        search_and_velocity_boundaries=benchmark_function['search_and_velocity_boundaries'],
        # wt=WallTypes.ELIMINATING
    )

    eis_l2_k0p2c_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=benchmark_function['formula'],
        spawn_boundaries=benchmark_function['search_domain'],
        objective_function_goal_point=benchmark_function['goal_point'],
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans*particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.CONSTANT, 0.2)),
        search_and_velocity_boundaries=benchmark_function['search_and_velocity_boundaries'],
        # wt=WallTypes.ELIMINATING
    )

    eis_l2_k1c_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=benchmark_function['formula'],
        spawn_boundaries=benchmark_function['search_domain'],
        objective_function_goal_point=benchmark_function['goal_point'],
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.CONSTANT, 1.0)),
        search_and_velocity_boundaries=benchmark_function['search_and_velocity_boundaries'],
        # wt=WallTypes.ELIMINATING
    )

    eis_l2_k2c_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=benchmark_function['formula'],
        spawn_boundaries=benchmark_function['search_domain'],
        objective_function_goal_point=benchmark_function['goal_point'],
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.CONSTANT, 2.0)),
        search_and_velocity_boundaries=benchmark_function['search_and_velocity_boundaries'],
        # wt=WallTypes.ELIMINATING
    )

    eis_l2_k0p2l_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=benchmark_function['formula'],
        spawn_boundaries=benchmark_function['search_domain'],
        objective_function_goal_point=benchmark_function['goal_point'],
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.LINEAR, 0.2)),
        search_and_velocity_boundaries=benchmark_function['search_and_velocity_boundaries'],
        # wt=WallTypes.ELIMINATING
    )

    eis_l2_k1l_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=benchmark_function['formula'],
        spawn_boundaries=benchmark_function['search_domain'],
        objective_function_goal_point=benchmark_function['goal_point'],
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.LINEAR, 1.0)),
        search_and_velocity_boundaries=benchmark_function['search_and_velocity_boundaries'],
        # wt=WallTypes.ELIMINATING
    )

    eis_l2_k0p2a_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=benchmark_function['formula'],
        spawn_boundaries=benchmark_function['search_domain'],
        objective_function_goal_point=benchmark_function['goal_point'],
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.ADAPTIVE, 0.2)),
        search_and_velocity_boundaries=benchmark_function['search_and_velocity_boundaries'],
        # wt=WallTypes.ELIMINATING
    )

    eis_l2_k1a_AClanPSO = scripts.experiments.experiments_data_creation.run(
        executor=executor,
        num_of_experiments=experiments,
        objective_function_pointer=benchmark_function['formula'],
        spawn_boundaries=benchmark_function['search_domain'],
        objective_function_goal_point=benchmark_function['goal_point'],
        maximum_iterations=maximum_iterations,
        swarm_size=number_of_clans * particles_per_clan,
        isClan=True,
        number_of_clans=number_of_clans,
        adaptivePSO=True,
        eis=((GlobalLocalCoefficientTypes.LINEAR, 2.0), (ControlFactorTypes.ADAPTIVE, 1.0)),
        search_and_velocity_boundaries=benchmark_function['search_and_velocity_boundaries'],
        # wt=WallTypes.ELIMINATING
    )



    sphere_collected_data = {
        "Simple AClanPSO": pandas.Series([simple_AClanPSO["Precision mean"], simple_AClanPSO["Precision std"], simple_AClanPSO["Precision median"],
                                         simple_AClanPSO["Best precision"],
                                          simple_AClanPSO["Mean iterations per experiment"], maximum_iterations,
                                         simple_AClanPSO["Mean iteration CPU time"], simple_AClanPSO["Mean experiment CPU time"],
                                         experiments,
                                          number_of_clans, particles_per_clan],
                                        index=["Precision mean",
                                               "Precision std",
                                               "Precision median",
                                               "Best precision",
                                               "Mean iterations per experiment",
                                               "Maximum iterations per experiment",
                                               "Mean iteration CPU time",
                                               "Mean experiment CPU time",
                                               "Experiments",
                                               "No. clans",
                                               "Particles per clan"]),

        "EIS-L2-k0.2C AClanPSO": pandas.Series(
            [eis_l2_k0p2c_AClanPSO["Precision mean"], eis_l2_k0p2c_AClanPSO["Precision std"],
             eis_l2_k0p2c_AClanPSO["Precision median"], eis_l2_k0p2c_AClanPSO["Best precision"],
             eis_l2_k0p2c_AClanPSO["Mean iterations per experiment"], maximum_iterations,
             eis_l2_k0p2c_AClanPSO["Mean iteration CPU time"], eis_l2_k0p2c_AClanPSO["Mean experiment CPU time"],
             experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Maximum iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]),

        "EIS-L2-k1C AClanPSO": pandas.Series(
            [eis_l2_k1c_AClanPSO["Precision mean"], eis_l2_k1c_AClanPSO["Precision std"], eis_l2_k1c_AClanPSO["Precision median"],
             eis_l2_k1c_AClanPSO["Best precision"],
             eis_l2_k1c_AClanPSO["Mean iterations per experiment"], maximum_iterations,
             eis_l2_k1c_AClanPSO["Mean iteration CPU time"], eis_l2_k1c_AClanPSO["Mean experiment CPU time"],
             experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Maximum iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]),

        "EIS-L2-k2C AClanPSO": pandas.Series(
            [eis_l2_k2c_AClanPSO["Precision mean"], eis_l2_k2c_AClanPSO["Precision std"],
             eis_l2_k2c_AClanPSO["Precision median"], eis_l2_k2c_AClanPSO["Best precision"],
             eis_l2_k2c_AClanPSO["Mean iterations per experiment"], maximum_iterations,
             eis_l2_k2c_AClanPSO["Mean iteration CPU time"], eis_l2_k2c_AClanPSO["Mean experiment CPU time"],
             experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Maximum iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]),

        "EIS-L2-k0.2L AClanPSO": pandas.Series(
            [eis_l2_k0p2l_AClanPSO["Precision mean"], eis_l2_k0p2l_AClanPSO["Precision std"],
             eis_l2_k0p2l_AClanPSO["Precision median"], eis_l2_k0p2l_AClanPSO["Best precision"],
             eis_l2_k0p2l_AClanPSO["Mean iterations per experiment"], maximum_iterations,
             eis_l2_k0p2l_AClanPSO["Mean iteration CPU time"], eis_l2_k0p2l_AClanPSO["Mean experiment CPU time"],
             experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Maximum iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]
        ),

        "EIS-L2-k1L AClanPSO": pandas.Series(
            [eis_l2_k1l_AClanPSO["Precision mean"], eis_l2_k1l_AClanPSO["Precision std"],
             eis_l2_k1l_AClanPSO["Precision median"], eis_l2_k1l_AClanPSO["Best precision"],
             eis_l2_k1l_AClanPSO["Mean iterations per experiment"], maximum_iterations,
             eis_l2_k1l_AClanPSO["Mean iteration CPU time"], eis_l2_k1l_AClanPSO["Mean experiment CPU time"],
             experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Maximum iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]
        ),

        "EIS-L2-k0.2A AClanPSO": pandas.Series(
            [eis_l2_k0p2a_AClanPSO["Precision mean"], eis_l2_k0p2a_AClanPSO["Precision std"],
             eis_l2_k0p2a_AClanPSO["Precision median"],
             eis_l2_k0p2a_AClanPSO["Best precision"],
             eis_l2_k0p2a_AClanPSO["Mean iterations per experiment"], maximum_iterations,
             eis_l2_k0p2a_AClanPSO["Mean iteration CPU time"], eis_l2_k0p2a_AClanPSO["Mean experiment CPU time"],
             experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Maximum iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]
        ),

        "EIS-L2-k1A AClanPSO": pandas.Series(
            [eis_l2_k1a_AClanPSO["Precision mean"], eis_l2_k1a_AClanPSO["Precision std"],
             eis_l2_k1a_AClanPSO["Precision median"], eis_l2_k1a_AClanPSO["Best precision"],
             eis_l2_k1a_AClanPSO["Mean iterations per experiment"], maximum_iterations,
             eis_l2_k1a_AClanPSO["Mean iteration CPU time"], eis_l2_k1a_AClanPSO["Mean experiment CPU time"],
             experiments,
             number_of_clans, particles_per_clan],
            index=["Precision mean",
                   "Precision std",
                   "Precision median",
                   "Best precision",
                   "Mean iterations per experiment",
                   "Maximum iterations per experiment",
                   "Mean iteration CPU time",
                   "Mean experiment CPU time",
                   "Experiments",
                   "No. clans",
                   "Particles per clan"]
        )
    }

    pandas.DataFrame(sphere_collected_data).to_csv(f'alpinen1_{number_of_clans}x{particles_per_clan}_without_ELS.csv')
    print(f'File for Alpine N.1 benchmark function with {number_of_clans} clans of {particles_per_clan} particles was created!')




if __name__ == "__main__":
    main()
