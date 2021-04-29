"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from numpy import absolute, sum, sqrt, square, sin, cos, prod, pi, e, array
from scipy.linalg import norm


# Theorem: min g(x) = -max (-g(x)).
# PSO traditionally follows the maximization problem,
# and it is for that reason that this approach has been followed in this implementation as well.


# Unimodal functions:
# -------------------

# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
def sphere_function(x: array) -> float:
    return -sum(square(x))


# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
def quadric_function(x: array) -> float:
    return -(sum(sum(x[j] for j in range(i+1))**2 for i in range(len(x))))


# Min: f(0,0,...,0) = 0, where 0 is repeated n times, where n is the domain dimensions.
# The function can be defined on any input domain but it is usually evaluated on xi∈[−100,100]for i=1,…,n
def schwefel222_function(x: array) -> float:
    return -(sum(absolute(x)) + prod(absolute(x)))


# Min: f(1,1,...,1) = 0  , where 1 is repeated n times, where n is the domain dimensions.
def rosenbrock_function(x: array, problem: bool = False) -> float:
    sum = 0
    for i in range(len(x)-1):
        sum += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
    if problem:
        return sum  # Maximization problem
    else:
        return 1/sum  # Minimization problem (default)



# Multimodal functions
# --------------------

# Because the Rastrigin function is a non-negative function (with f(x) = 0 <=> x = 0),
# instead of utilizing the theorem above for solving the minimalization problem,
# the inverse function is used as the objective function instead.
def rastrigin_function(x: array) -> float:
    A = 10
    return 1/(A * len(x) + sum(square(x) - A * cos(2*pi*x)))
    # return 1/(A * len(x) + sum(x[i]**2 - A * cos(2*pi*x[i]) for i in range(len(x))))


# Ackley function's domain is x[i] ∈ [-5, 5] for all i and its minimum is at f(0,...0) = 0.
def ackley_function(x: array, a: float = 20, b: float = 0.2, c: float = 2*pi) -> float:
    return -(-a *
             e ** (-b * (sqrt(1 / len(x) * sum(square(x)))))
             - e ** (1 / len(x) * sum(cos(c * x)))
             + a + e)


def salomon_function(x: array) -> float:
    norm_of_x = norm(x)
    return -(1 - cos(2*pi*norm_of_x) + 0.1*norm_of_x)


def alpinen1_function(x: array) -> float:
    return -sum(absolute(x * sin(x) + 0.1 * x))

def styblinski_tang_function(x: list) -> float:
    return -(sum(x[i]**4 -16*x[i]**2 + 5*x[i] for i in range(len(x)))/2)