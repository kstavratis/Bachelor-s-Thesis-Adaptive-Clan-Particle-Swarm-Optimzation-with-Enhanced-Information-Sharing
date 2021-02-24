from classes.classicPSOswarm import ClassicSwarm
from classes.clanPSOswarm import ClanSwarm
from classes.particle import Particle
from numpy import cos,pi, e, sqrt, inf


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



def main():
    domain_dimensions = 10


    rastrigin_function_search_domain = [[-5.12, 5.12] for i in range(domain_dimensions)]
    ackley_function_search_domain = [[-32.768, 32.768] for i in range(domain_dimensions)]
    ackley_function_search_domain = [[-5, 5] for i in range(domain_dimensions)]
    # sphere_function_search_domain = [[-inf, inf] for i in range(domain_dimensions)]
    sphere_function_search_domain = [[-10**2, 10**2] for i in range(domain_dimensions)]
    # The domain of Rosenbrock's function is x[i] ∈ [-∞, ∞] for all i
    # rosenbrock_function_search_domain = [[-inf, inf] for i in range(domain_dimensions)]
    rosenbrock_function_search_domain = [[-10**2, 10**2] for i in range(domain_dimensions)]
    styblinski_tang_function_search_domain = [[-5, 5] for i in range(domain_dimensions)]

    # To achieve a balance between global and local exploration to speed up convergence to the true optimum,
    # an inertia weight whose value decreases linearly with the iteration number has been used.
    # The values of w_min = 0.4 and w_max = 0.9 are widely used.
    w_min, w_max = 0.4, 0.9

    repetitions = 5000
    loop_stop_condition_limit = 5 * 10**(-7)

    print("Classic Particle Swarm Optimization: Rastrigin Function")
    print("-------------------------------------------------------")
    rastrigin_classic_swarm = ClassicSwarm(rastrigin_function, rastrigin_function_search_domain, w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not(loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
        # Linear decrease of (velocity) inertia weight.
        Particle.w = w_max - ((w_max - w_min)/repetitions) * iteration
        rastrigin_classic_swarm.update_swarm()
        loop_stop_condition_value = rastrigin_classic_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Rastrigin function optimizing point x = " + str(rastrigin_classic_swarm.global_best_position))
    print("Rastrigin value = " + str(rastrigin_function(rastrigin_classic_swarm.global_best_position)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("----------------------------------------------------")

    [print() for i in range(2)]

    print("Clan Particle Swarm Optimization: Rastrigin Function")
    print("----------------------------------------------------")
    rastrigin_clan_swarm = ClanSwarm(rastrigin_function, rastrigin_function_search_domain, w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
        # Linear decrease of (velocity) inertia weight.
        Particle.w = w_max - ((w_max - w_min) / repetitions) * iteration
        rastrigin_clan_swarm.update_swarm()
        loop_stop_condition_value = rastrigin_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Rastrigin function optimizing point x = " + str(rastrigin_clan_swarm.find_population_global_best_position()))
    print("Rastrigin value = " + str(rastrigin_function(rastrigin_clan_swarm.find_population_global_best_position())))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("----------------------------------------------------")



    [print() for i in range(6)]





    print("Classic Particle Swarm Optimization: Ackley Function")
    print("----------------------------------------------------")
    ackley_classic_swarm = ClassicSwarm(ackley_function, ackley_function_search_domain, w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not(loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
        Particle.w = w_max - ((w_max - w_min) / repetitions) * iteration
        ackley_classic_swarm.update_swarm()
        loop_stop_condition_value = ackley_classic_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Ackley function optimizing point x = " + str(ackley_classic_swarm.global_best_position))
    print("Ackley value = " + str(ackley_function((ackley_classic_swarm.global_best_position))))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("----------------------------------------------------")

    [print() for i in range(2)]

    print("Clan Particle Swarm Optimization: Ackley Function")
    print("-------------------------------------------------")
    ackley_clan_swarm = ClanSwarm(ackley_function, ackley_function_search_domain, w=w_max, swarm_size=50)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
        Particle.w = w_max - ((w_max - w_min) / repetitions) * iteration
        ackley_clan_swarm.update_swarm()
        loop_stop_condition_value = ackley_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Ackley function optimizing point x = " + str(ackley_clan_swarm.find_population_global_best_position()))
    print("Ackley value = " + str(ackley_function((ackley_clan_swarm.find_population_global_best_position()))))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("--------------------------------------------------")



    [print() for i in range(6)]




    print("Classic Particle Swarm Optimization: Sphere Function")
    print("----------------------------------------------------")
    sphere_classic_swarm = ClassicSwarm(sphere_function, sphere_function_search_domain, w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
        Particle.w = w_max - ((w_max - w_min) / repetitions) * iteration
        sphere_classic_swarm.update_swarm()
        sphere_classic_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Sphere function optimizing point x = " + str(sphere_classic_swarm.global_best_position))
    print("Sphere function value = " + str(sphere_function(sphere_classic_swarm.global_best_position)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("-------------------------------------------------")

    [print() for i in range(2)]

    print("Clan Particle Swarm Optimization: Sphere Function")
    print("----------------------------------------------------")
    sphere_clan_swarm = ClanSwarm(sphere_function, sphere_function_search_domain, w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
        Particle.w = w_max - ((w_max - w_min) / repetitions) * iteration
        sphere_clan_swarm.update_swarm()
        sphere_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Sphere function optimizing point x = " + str(sphere_clan_swarm.find_population_global_best_position()))
    print("Sphere function value = " + str(sphere_function(sphere_clan_swarm.find_population_global_best_position())))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("-------------------------------------------------")



    [print() for i in range(6)]




    print("Classic Particle Swarm Optimization: Rosenbrock Function")
    print("--------------------------------------------------------")
    rosenbrock_classic_swarm = ClassicSwarm(rosenbrock_function, rosenbrock_function_search_domain, w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
        Particle.w = w_max - ((w_max - w_min) / repetitions) * iteration
        rosenbrock_classic_swarm.update_swarm()
        rosenbrock_classic_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Rosenbrock function optimizing point x = " + str(rosenbrock_classic_swarm.global_best_position))
    print("Rosenbrock function value at x = " + str(rosenbrock_function(rosenbrock_classic_swarm.global_best_position)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("-----------------------------------------------------")

    [print() for i in range(2)]

    print("Clan Particle Swarm Optimization: Rosenbrock Function")
    print("--------------------------------------------------------")
    rosenbrock_clan_swarm = ClanSwarm(rosenbrock_function, rosenbrock_function_search_domain, w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
        Particle.w = w_max - ((w_max - w_min) / repetitions) * iteration
        rosenbrock_clan_swarm.update_swarm()
        rosenbrock_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Rosenbrock function optimizing point x = " + str(rosenbrock_clan_swarm.find_population_global_best_position()))
    print("Rosenbrock function value at x = " + str(rosenbrock_function(rosenbrock_clan_swarm.find_population_global_best_position())))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("-----------------------------------------------------")




    # styblinski_tang_swarm = ClassicSwarm(styblinski_tang_function, styblinski_tang_function_search_domain, w=w_max)
    # loop_stop_condition_value = inf
    # iteration = 0  # Counter used for changing inertia constants.
    # while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < repetitions:
    #     Particle.w = w_max - ((w_max - w_min) / repetitions) * iteration
    #     rosenbrock_swarm.update_swarm()
    #     rosenbrock_swarm.calculate_swarm_distance_from_swarm_centroid()
    #     iteration += 1
    # print("Rosenbrock function optimizing point x = " + str(rosenbrock_swarm.fitness_function[0].global_best_position))
    # print("iterations = " + str(iteration))
    # print("Stop condition value = " + str(loop_stop_condition_value))


if __name__ == "__main__":
    main()