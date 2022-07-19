"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from numpy import absolute, sum as npsum, sqrt, square, sin, cos, prod, pi, e, array, zeros, ones
from scipy.linalg import norm



# Theorem: min g(x) = -max (-g(x)).
# PSO traditionally follows the maximization problem,
# and it is for that reason that this approach has been followed in this implementation as well.


# Unimodal functions:
# -------------------

# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
def sphere_function_formula(x: array) -> float:
    return npsum(square(x))


# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
def quadric_function_formula(x: array) -> float:
    return sum(sum(x[j] for j in range(i+1))**2 for i in range(len(x)))


# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
# The function can be defined on any input domain but it is usually evaluated on xi∈[−100,100]for i=1,…,n
def schwefel222_function_formula(x: array) -> float:
    return npsum(absolute(x)) + prod(absolute(x))


# Min: f(1,1,...,1) = 0  , where 1 is repeated n times, where n is the domain dimensions.
def rosenbrock_function_formula(x: array, problem: bool = True) -> float:
    sum = 0
    for i in range(len(x)-1):
        sum += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
    if problem:
        return sum  # Minimization problem (default)
    else:
        return 1/sum  # Maximization problem



# Multimodal functions
# --------------------

# Because the Rastrigin function is a non-negative function (with f(x) = 0 <=> x = 0),
# instead of utilizing the theorem above for solving the minimalization problem,
# the inverse function is used as the objective function instead.
def rastrigin_function_formula(x: array) -> float:
    A = 10
    return A * len(x) + npsum(square(x) - A * cos(2*pi*x))
    # return 1/(A * len(x) + sum(x[i]**2 - A * cos(2*pi*x[i]) for i in range(len(x))))


# Ackley function's domain is x[i] ∈ [-5, 5] for all i and its minimum is at f(0,...0) = 0.
def ackley_function_formula(x: array, a: float = 20, b: float = 0.2, c: float = 2*pi) -> float:
    return -a *\
             e ** (-b * (sqrt(1 / len(x) * npsum(square(x)))))\
             - e ** (1 / len(x) * npsum(cos(c * x)))\
             + a + e


def salomon_function_formula(x: array) -> float:
    norm_of_x = norm(x)
    return 1 - cos(2*pi*norm_of_x) + 0.1*norm_of_x


def alpinen1_function_formula(x: array) -> float:
    return npsum(absolute(x * sin(x) + 0.1 * x))

def styblinski_tang_function_formula(x: list) -> float:
    return npsum(x[i]**4 -16*x[i]**2 + 5*x[i] for i in range(len(x)))/2





domain_dimensions = 10



# Defining dictionaries containing crucial information for each benchmark function.
# This is for code readability, as well as reusability in 'main.py' file.
sphere_function = {
    'formula': sphere_function_formula,
    'search_domain': [[-10 ** 2, 10 ** 2] for _ in range(domain_dimensions)],
    'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
    'goal_point': zeros(domain_dimensions)
}

quadric_function = {
    'formula': quadric_function_formula,
    'search_domain': [[-10 ** 2, 10 ** 2] for _ in range(domain_dimensions)],
    'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
    'goal_point': zeros(domain_dimensions)
}

schwefel222_function = {
    'formula': schwefel222_function_formula,
    'search_domain': [[-10, 10] for _ in range(domain_dimensions)],
    'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
    'goal_point': zeros(domain_dimensions)
}

rosenbrock_function = {
    'formula': rosenbrock_function_formula,
    'search_domain': [[-10, 10] for _ in range(domain_dimensions)],
    'search_and_velocity_boundaries': [[-10, 10], [-0.2 * 10, 0.2 * 10]],
    'goal_point': ones(domain_dimensions)
}

rastrigin_function = {
    'formula': rastrigin_function_formula,
    'search_domain': [[-5.12, 5.12] for _ in range(domain_dimensions)],
    'search_and_velocity_boundaries': [[-5.12, 5.12], [-0.2 * 5.12, 0.2 * 5.12]],
    'goal_point': zeros(domain_dimensions)
}

ackley_function = {
    'formula': ackley_function_formula,
    'search_domain': [[-32, 32] for _ in range(domain_dimensions)],
    'search_and_velocity_boundaries': [[-32, 32], [-0.2 * 32, 0.2 * 32]],
    'goal_point': zeros(domain_dimensions)
}

salomon_function = {
    'formula': salomon_function_formula,
    'search_domain': [[-10**2, 10**2] for _ in range(domain_dimensions)],
    'search_and_velocity_boundaries': [[-100, 100], [-0.2 * 100, 0.2 * 100]],
    'goal_point': zeros(domain_dimensions)
}

alpinen1_function = {
    'formula': alpinen1_function_formula,
    'search_domain': [[0, 10] for _ in range(domain_dimensions)],
    'search_and_velocity_boundaries': [[0, 10], [-0.2 * 10, 0.2 * 10]],
    'goal_point': zeros(domain_dimensions)
}
