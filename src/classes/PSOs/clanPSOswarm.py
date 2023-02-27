"""
Copyright (C) 2022  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""


from .classicPSOswarm import ClassicSwarm
from .wall_types import WallTypes
from .Mixins.enhanced_information_sharing.enums.global_local_coefficient_types import GlobalLocalCoefficientTypes
from .Mixins.enhanced_information_sharing.enums.control_factor_types import ControlFactorTypes
from numpy import array_equal
from typing import List, Tuple



class ClanSwarm:
    def __init__(self, fitness_function, spawn_boundaries: list,
                 maximum_iterations,
                 swarm_size: int = 15, number_of_clans: int = 4,
                 c1: float = 2, c2: float = 2,
                 is_adaptive: bool = False,
                 eis: Tuple[Tuple[GlobalLocalCoefficientTypes, float or None], Tuple[ControlFactorTypes, float or None]] =
                 ((GlobalLocalCoefficientTypes.NONE, None), (ControlFactorTypes.NONE, None)),
                 current_iteration: int = 0,
                 search_and_velocity_boundaries: List[List[float]] = None, wt: WallTypes = WallTypes.NONE):

        self.clans = [ClassicSwarm(fitness_function, spawn_boundaries, is_adaptive=is_adaptive,
                                   maximum_iterations=maximum_iterations,
                                   # w=w,
                                   c1=c1, c2=c2,
                                   eis=eis,
                                   swarm_size=swarm_size, current_iteration=current_iteration,
                                   search_and_velocity_boundaries=search_and_velocity_boundaries, wt=wt)
                      for i in range(number_of_clans)]
        self.__fitness_function = fitness_function


        self.__max_iterations = maximum_iterations
        self.__spawn_boundaries = spawn_boundaries
        self.__search_and_velocity_boundaries, self.__wall_type = search_and_velocity_boundaries, wt

    def update_swarm(self):
        def find_clan_leaders():
            leaders = []
            for swarm in self.clans:
                for particle in swarm.swarm:
                    # "For each iteration, each clan performs a search using a common PSO and marks the particle
                    # that **had reached** the best position of the entire clan.
                    # The marked particle is called the leader and this process is called delegation."
                    # ~Adaptive Clan Particle Optimization II: CLAN PARTICLE SWARM OPTIMIZATION 2nd paragraph.
                    if array_equal(swarm.global_best_position, particle._personal_best_position):
                        leaders.append(particle)
                        break
            return leaders

        def update_clan_leaders(leaders: list):
            clan_leaders_swarm = ClassicSwarm(
                leaders,
                spawn_boundaries=self.__spawn_boundaries,
                maximum_iterations=self.__max_iterations,
                current_iteration=self.clans[0].current_iteration,  # Note: all clans have the same value in their "current_iteration" variable.
                eis=((self.clans[0]._global_local_coefficient_method, self.clans[0].c3),
                     (self.clans[0]._control_factor_method, self.clans[0].c3_k)),
                search_and_velocity_boundaries=self.__search_and_velocity_boundaries, wt=self.__wall_type
            )

            clan_leaders_swarm.update_swarm()

        # Execute particle movement as per the classic PSO
        for swarm in self.clans:
            swarm.update_swarm()
        # Execute Clan PSO movement operation.
        update_clan_leaders(find_clan_leaders())

    # TODO
    #   While the implementation below is computationally correct, it has one main weakness:
    #   it recalculates the fitness value for each point, which may be computationally expensive,
    #   such as in the case of the Quadric function, where the cost of computing a value is O(n^2).
    #   This principle may be generalized into other computationally expensive problems (fitness function values),
    #   such as calculating the error of a neural network training cycle.
    #   POSSIBLE SOLUTION: Store the best particle population at runtime, using either a pointer to the
    #   particle object or an index to its position in the array/list.
    def find_population_global_best_position(self):
        # Essentially, what is done:
        # Step 1) Find best particle in each clan (inner 'min' function).
        # Step 2) Among the clan leaders calculated in the previous step, find the best particle among them.
        #! Step 2 error!
        #! Finds the best position at the CURRENT CONFIGURATION of the each swarm.
        #! However, it may be the case that the global best of the swarm is not included
        #! in (at least one) swarm considered in the (inner) "min" function! 
        # 'min' function is used because the implementation follows the "better => lower" convention.
        # Programming note: Using python built-in function increases performance, as the functions run in C internally,
        # while "for" loops in Python are a while-try-catch loop, which is computationally more expensive.
        return min([self.clans[i].f_global_best for i in range(len(self.clans))])

    def calculate_swarm_distance_from_swarm_centroid(self):
        return 1/len(self.clans) *\
               sum(clan.calculate_swarm_distance_from_swarm_centroid()
                                       for clan in self.clans)