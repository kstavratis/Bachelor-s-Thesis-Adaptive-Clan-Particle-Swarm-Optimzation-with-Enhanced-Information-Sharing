"""
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

import numpy as np

from . import formulae_definitions as fd

domain_dimensions = 30

# ==================== Benchmark functions constructors START ====================

__functions_constructors_list = []
# Defining dictionaries containing crucial information for each benchmark function.
# This is for code readability, as well as reusability in 'main.py' file.

sphere : dict = None
def __sphere_function_constructor() -> dict:
    global sphere
    sphere = {
       'name' : 'sphere',
        'formula': fd.sphere_formula,
        'search_domain': np.tile(np.array([-10 ** 2, 10 ** 2]), (domain_dimensions, 1)),
        'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
        'goal_point': np.zeros(domain_dimensions) 
    }
__functions_constructors_list.append(__sphere_function_constructor)

quadric : dict = None
def __quadric_function_constructor() -> dict:
    global quadric
    quadric = {
        'name' : 'quadric',
        'formula': fd.quadric_formula,
        'search_domain': np.tile(np.array([-10 ** 2, 10 ** 2]), (domain_dimensions, 1)),
        'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
        'goal_point': np.zeros(domain_dimensions)
    }
__functions_constructors_list.append(__quadric_function_constructor)

schwefel222 : dict = None
def __schwefel222_function_constructor() -> dict:
    global schwefel222
    schwefel222 = {
        'name' : 'schwefel222',
        'formula': fd.schwefel222_formula,
        'search_domain': np.tile(np.array([-10, 10]), (domain_dimensions, 1)),
        'search_and_velocity_boundaries': [[-10, 10], [-0.2 * 10, 0.2 * 10]],
        'goal_point': np.zeros(domain_dimensions)
    }
__functions_constructors_list.append(__schwefel222_function_constructor)


rosenbrock : dict = None
def __rosenbrock_function_constructor() -> dict:
    global rosenbrock
    rosenbrock = {
        'name' : 'rosenbrock',
        'formula': fd.rosenbrock_formula,
        'search_domain': np.tile(np.array([-10, 10]), (domain_dimensions, 1)),
        'search_and_velocity_boundaries': [[-10, 10], [-0.2 * 10, 0.2 * 10]],
        'goal_point': np.ones(domain_dimensions)
    }
__functions_constructors_list.append(__rosenbrock_function_constructor)

rastrigin : dict = None
def __rastrigin_function_constructor() -> dict:
    global rastrigin
    rastrigin = {
        'name' : 'rastrigin',
        'formula': fd.rastrigin_formula,
        'search_domain': np.tile(np.array([-5.12, 5.12]), (domain_dimensions, 1)),
        'search_and_velocity_boundaries': [[-5.12, 5.12], [-0.2 * 5.12, 0.2 * 5.12]],
        'goal_point': np.zeros(domain_dimensions)
    }
__functions_constructors_list.append(__rastrigin_function_constructor)

ackley : dict = None
def __ackley_function_constructor() -> dict:
    global ackley
    ackley = {
        'name' : 'ackley',
        'formula': fd.ackley_formula,
        'search_domain': np.tile(np.array([-32, 32]), (domain_dimensions, 1)),
        'search_and_velocity_boundaries': [[-32, 32], [-0.2 * 32, 0.2 * 32]],
        'goal_point': np.zeros(domain_dimensions)
    }
__functions_constructors_list.append(__ackley_function_constructor)

salomon : dict = None
def __salomon_function_constructor() -> dict:
    global salomon
    salomon = {
        'name' : 'salomon',
        'formula': fd.salomon_formula,
        'search_domain': np.tile(np.array([-10**2, 10**2]), (domain_dimensions, 1)),
        'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
        'goal_point': np.zeros(domain_dimensions)
    }
__functions_constructors_list.append(__salomon_function_constructor)

alpinen1 : dict = None
def __alpinen1_function_constructor() -> dict:
    global alpinen1
    alpinen1 = {
        'name' : 'alpinen1',
        'formula': fd.alpinen1_formula,
        'search_domain': np.tile(np.array([0, 10]), (domain_dimensions, 1)),
        'search_and_velocity_boundaries': [[0, 10], [-0.2 * 10, 0.2 * 10]],
        'goal_point': np.zeros(domain_dimensions)
    }
__functions_constructors_list.append(__alpinen1_function_constructor)

# ==================== Benchmark functions constructors FINISH ====================


def reset_benchmark_functions_dictionaries():
    for fcn_ptr in __functions_constructors_list:
        fcn_ptr()

# First call of the function, so as to initialize the benchmark functions' information.
reset_benchmark_functions_dictionaries()