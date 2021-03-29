from classes.PSOs.classicPSOswarm import ClassicSwarm
from classes.PSOs.clanPSOswarm import ClanSwarm
from classes.wall_types import WallTypes
from numpy import cos,pi, e, sqrt, inf, array as vector, seterr
from numpy.linalg import norm
import matplotlib.pyplot as plt



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
    # seterr(all='raise')
    domain_dimensions = 10

    fig, axis = plt.subplots()


    rastrigin_function_search_domain = [[-5.12, 5.12] for i in range(domain_dimensions)]
    ackley_function_search_domain = [[-32.768, 32.768] for i in range(domain_dimensions)]
    ackley_function_search_domain = [[-5, 5] for i in range(domain_dimensions)]
    # sphere_function_search_domain = [[-inf, inf] for i in range(domain_dimensions)]
    sphere_function_search_domain = [[-10**2, 10**2] for i in range(domain_dimensions)]
    # The domain of Rosenbrock's function is x[i] ∈ [-∞, ∞] for all i
    # rosenbrock_function_search_domain = [[-inf, inf] for i in range(domain_dimensions)]
    rosenbrock_function_search_domain = [[-10**2, 10**2] for i in range(domain_dimensions)]
    styblinski_tang_function_search_domain = [[-5, 5] for i in range(domain_dimensions)]


    rastigin_function_goal_point = vector([0 for i in range(domain_dimensions)])
    ackley_function_goal_point = vector([0 for i in range(domain_dimensions)])
    sphere_function_goal_point = vector([0 for i in range(domain_dimensions)])
    rosenbrock_function_goal_point = vector([1 for i in range(domain_dimensions)])

    # To achieve a balance between global and local exploration to speed up convergence to the true optimum,
    # an inertia weight whose value decreases linearly with the iteration number has been used.
    # The values of w_min = 0.4 and w_max = 0.9 are widely used.
    w_min, w_max = 0.4, 0.9


    maximum_iterations = 5000
    loop_stop_condition_limit = 5 * 10**(-20)

    print("Classic Particle Swarm Optimization: Rastrigin Function")
    print("-------------------------------------------------------")
    rastrigin_classic_divergences = []
    for i in range(1):  # Thirty (30) repetitions are considered enough for analysis/observations.
        rastrigin_classic_swarm = ClassicSwarm(rastrigin_function, rastrigin_function_search_domain, maximum_iterations,
                                               swarm_size=60, adaptive=True,
                                               search_and_velocity_boundaries=[[-5.12, 5.12], [-20 / 100 * 5.12, 20 / 100 * 5.12]],
                                               wt=WallTypes.ABSORBING)
        iteration = 0  # Counter used for changing inertia constants.
        loop_stop_condition_value = inf
        while not(loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
            rastrigin_classic_swarm.update_swarm()
            loop_stop_condition_value = rastrigin_classic_swarm.calculate_swarm_distance_from_swarm_centroid()
            iteration += 1
            # print(iteration)
        rastrigin_classic_divergences.append(norm(rastrigin_classic_swarm.global_best_position - rastigin_function_goal_point))

    print("Rastrigin function optimizing point x = " + str(rastrigin_classic_swarm.global_best_position))
    print("Rastrigin value = " + str(rastrigin_function(rastrigin_classic_swarm.global_best_position)))
    print("Distance from target = " + str(norm(rastrigin_classic_swarm.global_best_position - rastigin_function_goal_point)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("----------------------------------------------------")

    [print() for i in range(2)]

    print("Clan Particle Swarm Optimization: Rastrigin Function")
    print("----------------------------------------------------")
    rastrigin_clan_divergences = []
    for i in range(1):  # Thirty (30) repetitions are considered enough for analysis/observations.
        rastrigin_clan_swarm = ClanSwarm(rastrigin_function, rastrigin_function_search_domain, maximum_iterations,
                                         swarm_size=15, number_of_clans=4)
        iteration = 0  # Counter used for changing inertia constants.
        loop_stop_condition_value = inf
        while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
            rastrigin_clan_swarm.update_swarm()
            loop_stop_condition_value = rastrigin_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
            iteration += 1
        rastrigin_clan_divergences.append(norm(rastrigin_clan_swarm.find_population_global_best_position() - rastigin_function_goal_point))

    print("Rastrigin function optimizing point x = " + str(rastrigin_clan_swarm.find_population_global_best_position()))
    print("Rastrigin value = " + str(rastrigin_function(rastrigin_clan_swarm.find_population_global_best_position())))
    print("Distance from target = " + str(norm(rastrigin_clan_swarm.find_population_global_best_position() - rastigin_function_goal_point)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("----------------------------------------------------")


    axis.set_ylabel("Distance from target")
    axis.plot(rastrigin_classic_divergences, marker=".", label="Classic PSO")
    axis.plot(rastrigin_clan_divergences, marker=".", label="Clan PSO")
    # axis.plot(rastrigin_classic_iterations,rastrigin_classic_errors, label="Classic PSO")
    # axis.plot(rastrigin_clan_iterations, rastrigin_clan_errors, label="Clan PSO")
    axis.legend()
    plt.show()

    [print() for i in range(6)]





    print("Classic Particle Swarm Optimization: Ackley Function")
    print("----------------------------------------------------")
    ackley_classic_swarm = ClassicSwarm(ackley_function, ackley_function_search_domain, maximum_iterations)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not(loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
        ackley_classic_swarm.update_swarm()
        loop_stop_condition_value = ackley_classic_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Ackley function optimizing point x = " + str(ackley_classic_swarm.global_best_position))
    print("Ackley value = " + str(ackley_function((ackley_classic_swarm.global_best_position))))
    print("Distance from target = " + str(norm(ackley_classic_swarm.global_best_position - ackley_function_goal_point)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("----------------------------------------------------")

    [print() for i in range(2)]

    print("Clan Particle Swarm Optimization: Ackley Function")
    print("-------------------------------------------------")
    ackley_clan_swarm = ClanSwarm(ackley_function, ackley_function_search_domain, maximum_iterations, swarm_size=50)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
        ackley_clan_swarm.update_swarm()
        loop_stop_condition_value = ackley_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Ackley function optimizing point x = " + str(ackley_clan_swarm.find_population_global_best_position()))
    print("Ackley value = " + str(ackley_function((ackley_clan_swarm.find_population_global_best_position()))))
    print("Distance from target = " + str(norm(ackley_clan_swarm.find_population_global_best_position() - ackley_function_goal_point)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("--------------------------------------------------")



    [print() for i in range(6)]




    print("Classic Particle Swarm Optimization: Sphere Function")
    print("----------------------------------------------------")
    sphere_classic_swarm = ClassicSwarm(sphere_function, sphere_function_search_domain, maximum_iterations, adaptive=False)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
        sphere_classic_swarm.update_swarm()
        loop_stop_condition_value = sphere_classic_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Sphere function optimizing point x = " + str(sphere_classic_swarm.global_best_position))
    print("Sphere function value = " + str(sphere_function(sphere_classic_swarm.global_best_position)))
    print("Distance from target = " + str(norm(sphere_classic_swarm.global_best_position - sphere_function_goal_point)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("-------------------------------------------------")

    [print() for i in range(2)]

    print("Clan Particle Swarm Optimization: Sphere Function")
    print("----------------------------------------------------")
    sphere_clan_swarm = ClanSwarm(sphere_function, sphere_function_search_domain, maximum_iterations)
                                  # w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
        sphere_clan_swarm.update_swarm()
        loop_stop_condition_value = sphere_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Sphere function optimizing point x = " + str(sphere_clan_swarm.find_population_global_best_position()))
    print("Sphere function value = " + str(sphere_function(sphere_clan_swarm.find_population_global_best_position())))
    print("Distance from target = " + str(norm(sphere_clan_swarm.find_population_global_best_position() - sphere_function_goal_point)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("-------------------------------------------------")



    [print() for i in range(6)]




    print("Classic Particle Swarm Optimization: Rosenbrock Function")
    print("--------------------------------------------------------")
    rosenbrock_classic_swarm = ClassicSwarm(rosenbrock_function, rosenbrock_function_search_domain, maximum_iterations)
                                            # w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
        rosenbrock_classic_swarm.update_swarm()
        loop_stop_condition_value = rosenbrock_classic_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Rosenbrock function optimizing point x = " + str(rosenbrock_classic_swarm.global_best_position))
    print("Rosenbrock function value at x = " + str(rosenbrock_function(rosenbrock_classic_swarm.global_best_position)))
    print("Distance from target = " + str(norm(rosenbrock_classic_swarm.global_best_position - rosenbrock_function_goal_point)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("-----------------------------------------------------")

    [print() for i in range(2)]

    print("Clan Particle Swarm Optimization: Rosenbrock Function")
    print("--------------------------------------------------------")
    rosenbrock_clan_swarm = ClanSwarm(rosenbrock_function, rosenbrock_function_search_domain, maximum_iterations)
                                      # w=w_max)
    loop_stop_condition_value = inf
    iteration = 0  # Counter used for changing inertia constants.
    while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
        rosenbrock_clan_swarm.update_swarm()
        loop_stop_condition_value = rosenbrock_clan_swarm.calculate_swarm_distance_from_swarm_centroid()
        iteration += 1
    print("Rosenbrock function optimizing point x = " + str(rosenbrock_clan_swarm.find_population_global_best_position()))
    print("Rosenbrock function value at x = " + str(rosenbrock_function(rosenbrock_clan_swarm.find_population_global_best_position())))
    print("Distance from target = " + str(norm(rosenbrock_clan_swarm.find_population_global_best_position() - rosenbrock_function_goal_point)))
    print("iterations = " + str(iteration))
    print("Stop condition value = " + str(loop_stop_condition_value))
    print("-----------------------------------------------------")




    # styblinski_tang_swarm = ClassicSwarm(styblinski_tang_function, styblinski_tang_function_search_domain, w=w_max)
    # loop_stop_condition_value = inf
    # iteration = 0  # Counter used for changing inertia constants.
    # while not (loop_stop_condition_value < loop_stop_condition_limit) and iteration < maximum_iterations:
    #     Particle.w = w_max - ((w_max - w_min) / maximum_iterations) * iteration
    #     rosenbrock_swarm.update_swarm()
    #     rosenbrock_swarm.calculate_swarm_distance_from_swarm_centroid()
    #     iteration += 1
    # print("Rosenbrock function optimizing point x = " + str(rosenbrock_swarm.fitness_function[0].global_best_position))
    # print("iterations = " + str(iteration))
    # print("Stop condition value = " + str(loop_stop_condition_value))


if __name__ == "__main__":
    main()