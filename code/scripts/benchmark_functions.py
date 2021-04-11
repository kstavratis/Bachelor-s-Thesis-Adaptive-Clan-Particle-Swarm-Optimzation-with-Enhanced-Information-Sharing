"""
Copyright (C) 2021  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

from numpy import sqrt, cos, pi, e

# Theorem: min g(x) = -max (-g(x))


# Because the Rastrigin function is a non-negative function (with f(x) = 0 <=> x = 0),
# instead of utilizing the theorem above for solving the minimalization problem,
# the inverse function is used as the objective function instead.
def rastrigin_function(x: list):
    A = 10
    return 1/(A * len(x) + sum(x[i]**2 - A * cos(2*pi*x[i]) for i in range(len(x))))



# Ackley function's domain is x[i] ∈ [-5, 5] for all i and its minimum is at f(0,...0) = 0.
def ackley_function(x: list, a: float = 20, b: float = 0.2, c: float = 2*pi):
    return -(-a * \
           e**(-b * (sqrt(1/len(x) * sum(x[i]**2 for i in range(len(x)))))) \
           - e**(1/len(x) * sum(cos(c * x[i]) for i in range(len(x)))) \
           + a + e)


def sphere_function(x: list):
    return -(sum(x[i]**2 for i in range(len(x))))


# Min: f(1,1,...,1) = 0  , where 1 is repeated n times, where n is the domain dimentions.
def rosenbrock_function(x: list, problem: bool = False):
    sum = 0
    for i in range(len(x)-1):
        sum += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
    if problem:
        return sum  # Maximization problem
    else:
        return 1/sum  # Minimization problem (default)


def styblinski_tang_function(x: list):
    return -(sum(x[i]**4 -16*x[i]**2 + 5*x[i] for i in range(len(x)))/2)