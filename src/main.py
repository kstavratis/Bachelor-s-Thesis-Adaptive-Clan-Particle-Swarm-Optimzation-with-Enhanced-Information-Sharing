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

#import scripts.experiments.experiment
#import scripts.experiments.experiments_data_creation
import scripts.experiments.experiments_data_creation2
import scripts.experiments.experiments_animation_data_creation
import scripts.experiments.experimental_data_manipulation as data_avg
from classes.PSOs.Mixins.enhanced_information_sharing.enums.global_local_coefficient_types import GlobalLocalCoefficientTypes
from classes.PSOs.Mixins.enhanced_information_sharing.enums.control_factor_types import ControlFactorTypes


from numpy import seterr
import pandas as pd
from concurrent.futures.process import ProcessPoolExecutor

import scripts.benchmark_functions as bench_f


def main():
    #benchmark_function = bench_f.alpinen1_function
 
    number_of_clans = 6
    particles_per_clan = 5
    simple_pso_particles = number_of_clans * particles_per_clan
    maximum_iterations = 10
    experiments = 1
    executor = ProcessPoolExecutor()
 
    
    c3_initial_value = 2.0; c3_type = 'linear'
    c3k_initial_value = 1.0; c3k_type = 'linear'

    # benchmark_functions_list = [bench_f.ackley_function, bench_f.alpinen1_function, bench_f.quadric_function, 
    # bench_f.rastrigin_function, bench_f.rosenbrock_function, bench_f.schwefel222_function, bench_f.salomon_function, bench_f.sphere_function]

    benchmark_functions_list = [ bench_f.sphere_function]


    for benchmark_function in benchmark_functions_list:
        for c3k_initial_value in [1.0]:

            shared_keyword_input_parameters_dict = {
                    'objective_function_pointer' : benchmark_function['formula'] ,
                    'spawn_boundaries' : benchmark_function['search_domain'],
                    'objective_function_goal_point' : benchmark_function['goal_point'],
                    'maximum_iterations' : maximum_iterations,
                    'swarm_size' : number_of_clans * particles_per_clan,
                    'number_of_clans' : number_of_clans,
                    'search_and_velocity_boundaries' : benchmark_function['search_and_velocity_boundaries']
            }

            for mode in [ControlFactorTypes.CONSTANT]:

                eis_kL2_kcC02_tuple = ((GlobalLocalCoefficientTypes.LINEAR, c3_initial_value), (mode, c3k_initial_value))

                print(f'{benchmark_function["name"]} function: {number_of_clans} clans of {particles_per_clan} particles each')
                print("---------------")
                # EXECUTION TO COLLECT STATISTICAL DATA FOR MULTIPLE EXPERIMENTS
                # simple_AClanPSO_raw_data = scripts.experiments.experiments_data_creation2.run(
                #     executor=executor,
                #     num_of_experiments=experiments,
                #     kwargs = dict(**shared_keyword_input_parameters_dict, is_clan=False, is_adaptive=False)
                #     # kwargs = dict(**shared_keyword_input_parameters_dict, is_clan=False, is_adaptive=False, eis=eis_kL2_kcC02_tuple)

                #     # wt=WallTypes.ELIMINATING
                # )

                # COLLECT ANIMATION DATA FOR A SINGLE EXPERIMENT: ONE EXPERIMENT WITH MULTIPLE SNAPSHOTS
                simple_AClanPSO_raw_data = scripts.experiments.experiments_animation_data_creation.experiment(
                        kwargs=dict(**shared_keyword_input_parameters_dict, is_clan=False, is_adaptive=False),
                        # kwargs = dict(**shared_keyword_input_parameters_dict, is_clan=False, is_adaptive=False, eis=eis_kL2_kcC02_tuple)
                        )

                simple_AClanPSO_raw_data.to_csv(f'{benchmark_function["name"]}_function_eis_aclan_pso_'\
                    f'c3_{c3_type}_{c3_initial_value}_c3k_{"constant" if mode == ControlFactorTypes.CONSTANT else "linear" if mode == ControlFactorTypes.LINEAR else "adaptive"}'\
                    f'_{c3k_initial_value}_{number_of_clans}x{particles_per_clan}_raw_data.csv')

                # simple_AClanPSO_averaged_data = data_avg.distance_metric_mean(simple_AClanPSO_raw_data)
                # simple_AClanPSO_averaged_data.to_csv(f'{benchmark_function["name"]}_function_eis_aclan_pso_'\
                #     f'c3_{c3_type}_{c3_initial_value}_c3k_{"constant" if mode == ControlFactorTypes.ADAPTIVE else "linear" if mode == ControlFactorTypes.LINEAR else "adaptive"}'\
                #     f'_{c3k_initial_value}_{number_of_clans}x{particles_per_clan}_mean_of_experiments_per_iteration.csv')



if __name__ == "__main__":
    main()
