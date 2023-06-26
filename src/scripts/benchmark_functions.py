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

import numpy as np


# Theorem: min g(x) = -max (-g(x)).
# PSO traditionally follows the minimization problem formulation, which is standard in (mathematical) optimization.
# It is for this reason that this approach has been followed in this implementation as well.

domain_dimensions = 2

# Unimodal functions:
# -------------------

# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
def sphere_function_formula(x: np.array) -> float:
    """
    Arguments
    ---------
    x : np.array
        shape == (n, m)
        A 2D matrix of arbitrary (finite) non-negative shape.
    
    Returns
    -------
    : np.array
        shape == (n,) 
        The row-wise sphere function result
        # Math: f(\mathbf{v}) = \|\mathbf{v}\|^2, \dim(\mathbf{v})= D
    """
    return np.sum(np.square(x), axis=1)

# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
def quadric_function_formula(x: np.array) -> float:
    """
    Arguments
    ---------
    x : np.array
        shape == (n, m)
        A 2D matrix of arbitrary (finite) non-negative shape.

    Returns
    -------
     : np.array
        shape == (n,)
        The row-wise quadric function result
        # Math: f(\mathbf{v}) = Σ_i^D (Σ_j^i v_j)^2, \dim(\mathbf{v})= D
    """
    return np.sum(np.cumsum(x, axis=1)**2, axis=1)


# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
# The function can be defined on any input domain but it is usually evaluated on xi∈[−100,100]for i=1,…,n
def schwefel222_function_formula(x: np.array) -> float:
    """
    Arguments
    ---------
    x : np.array
        shape == (n,m)
        A 2D matrix of arbitrary (finite) non-negative shape.

    Returns
    -------
    : np.array
        shape == (n,)
        The row-wise Schwefel P2.22 function result
        # Math: f(\mathbf{v}) = Σ_i^D |v_i| + Π_i^D |v_i|, \dim(\mathbf{v})= D 
    """
    return np.sum(np.absolute(x), axis=1) + np.prod(np.absolute(x), axis=1)


# Min: f(1,1,...,1) = 0  , where 1 is repeated n times, where n is the domain dimensions.
def rosenbrock_function_formula(x: np.array) -> float:
    """
    Arguments
    ---------
    x : np.array
        shape == (n,m)
        A 2D matrix of arbitrary (finite) non-negative shape.

    Returns
    -------
    : np.array
        shape == (n,)
        The row-wise Rosenbrock function result
        # Math: f(\mathbf{v}) = \sum_i^{D-1} 100(v_{i+1} - v_i)^2 + (1 - v_i)^2, \dim(\mathbf{v}) = D  
    """
    xi_plus_1_xi = np.diff(x, n=1, axis=1)
    one_minus_xi = (np.ones_like(x) - x)[:, :-1]

    return np.sum(100*xi_plus_1_xi**2 + one_minus_xi**2, axis=1)



# Multimodal functions
# --------------------

# Because the Rastrigin function is a non-negative function (with f(x) = 0 <=> x = 0),
# instead of utilizing the theorem above for solving the minimalization problem,
# the inverse function is used as the objective function instead.
def rastrigin_function_formula(x: np.array) -> float:
    """
     Arguments
    ---------
    x : np.array
        shape == (n,m)
        A 2D matrix of arbitrary (finite) non-negative shape.

    Returns
    -------
    : np.array
        shape == (n,)
        The row-wise Rosenbrock function result
        # Math: f(\mathbf{v}) = 10 \cdot D  + \sum_i^{D} v_i^2 - 10 \cos(2 \pi v_i), \dim(\mathbf{v}) = D  
    """
    
    A = 10
    return A * x.shape[1] + np.sum(np.square(x) - A * np.cos(2*np.pi*x), axis=1)


# Ackley function's domain is x[i] ∈ [-5, 5] for all i and its minimum is at f(0,...0) = 0.
def ackley_function_formula(x: np.array, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> float:
    """
     Arguments
    ---------
    x : np.array
        shape == (n,m)
        A 2D matrix of arbitrary (finite) non-negative shape.

    Returns
    -------
    : np.array
        shape == (n,)
        The row-wise Ackley function result
        # Math: f(\mathbf{v}) = -20 \exp \left(-0.2 \sqrt{\frac{1}{D} \sum_{i=1}^{D} v_i^2} \right)\
        # Math: -\exp \left( \frac{1}{D} \sum_{i=1}^{D} \cos(2 \pi v_i) \right) + 20 + e, \dim(\mathbf{v}) = D  
    """
    return -a *\
             np.exp(-b * (np.sqrt(1 / x.shape[1] * np.sum(np.square(x), axis=1))))\
             - np.exp(1 / x.shape[1] * np.sum(np.cos(c * x), axis=1))\
             + a + np.e


def salomon_function_formula(x: np.array) -> float:
    """
     Arguments
    ---------
    x : np.array
        shape == (n,m)
        A 2D matrix of arbitrary (finite) non-negative shape.

    Returns
    -------
    : np.array
        shape == (n,)
        The row-wise Ackley function result
        # Math: f(\mathbf{v}) = 1 - \cos \left( 2 \pi \sqrt{\sum_{i=1}^{D} v_i^2} \right) +  0.1 \sqrt{\sum_{i=1}^{D}v_i^2}, \dim(\mathbf{v}) = D 
    """
    root_norm_of_x = np.sqrt(np.linalg.norm(x, axis=1))
    return 1 - np.cos(2*np.pi*root_norm_of_x) + 0.1*root_norm_of_x


def alpinen1_function_formula(x: np.array) -> float:
    """
     Arguments
    ---------
    x : np.array
        shape == (n,m)
        A 2D matrix of arbitrary (finite) non-negative shape.

    Returns
    -------
    : np.array
        shape == (n,)
        The row-wise Ackley function result
        # Math: f(\mathbf{v}) = \sum_{i=1}^{D} | v_i \sin(v_i) + 0.1 v_i |, \dim(\mathbf{v}) = D 
    """
    return np.sum(np.absolute(x * np.sin(x) + 0.1 * x), axis=1)

def styblinski_tang_function_formula(x: list) -> float:
    return np.sum(x[i]**4 -16*x[i]**2 + 5*x[i] for i in range(len(x)))/2







# Defining dictionaries containing crucial information for each benchmark function.
# This is for code readability, as well as reusability in 'main.py' file.
sphere_function = {
    'name' : 'sphere',
    'formula': sphere_function_formula,
    'search_domain': np.tile(np.array([-10 ** 2, 10 ** 2]), (domain_dimensions, 1)),
    'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
    'goal_point': np.zeros(domain_dimensions)
}

quadric_function = {
    'name' : 'quadric',
    'formula': quadric_function_formula,
    'search_domain': np.tile(np.array([-10 ** 2, 10 ** 2]), (domain_dimensions, 1)),
    'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
    'goal_point': np.zeros(domain_dimensions)
}

schwefel222_function = {
    'name' : 'schwefel222',
    'formula': schwefel222_function_formula,
    'search_domain': np.tile(np.array([-10, 10]), (domain_dimensions, 1)),
    'search_and_velocity_boundaries': [[-10, 10], [-0.2 * 10, 0.2 * 10]],
    'goal_point': np.zeros(domain_dimensions)
}

rosenbrock_function = {
    'name' : 'rosenbrock',
    'formula': rosenbrock_function_formula,
    'search_domain': np.tile(np.array([-10, 10]), (domain_dimensions, 1)),
    'search_and_velocity_boundaries': [[-10, 10], [-0.2 * 10, 0.2 * 10]],
    'goal_point': np.ones(domain_dimensions)
}

rastrigin_function = {
    'name' : 'rastrigin',
    'formula': rastrigin_function_formula,
    'search_domain': np.tile(np.array([-5.12, 5.12]), (domain_dimensions, 1)),
    'search_and_velocity_boundaries': [[-5.12, 5.12], [-0.2 * 5.12, 0.2 * 5.12]],
    'goal_point': np.zeros(domain_dimensions)
}

ackley_function = {
    'name' : 'ackley',
    'formula': ackley_function_formula,
    'search_domain': np.tile(np.array([-32, 32]), (domain_dimensions, 1)),
    'search_and_velocity_boundaries': [[-32, 32], [-0.2 * 32, 0.2 * 32]],
    'goal_point': np.zeros(domain_dimensions)
}

salomon_function = {
    'name' : 'salomon',
    'formula': salomon_function_formula,
    'search_domain': np.tile(np.array([-10**2, 10**2]), (domain_dimensions, 1)),
    'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
    'goal_point': np.zeros(domain_dimensions)
}

alpinen1_function = {
    'name' : 'alpinen1',
    'formula': alpinen1_function_formula,
    'search_domain': np.tile(np.array([0, 10]), (domain_dimensions, 1)),
    'search_and_velocity_boundaries': [[0, 10], [-0.2 * 10, 0.2 * 10]],
    'goal_point': np.zeros(domain_dimensions)
}
