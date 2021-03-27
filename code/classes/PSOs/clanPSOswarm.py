from classes.PSOs.classicPSOswarm import ClassicSwarm
from numpy import array_equal


class ClanSwarm:
    def __init__(self, fitness_function, convex_boundaries: list, maximum_iterations, adaptive: bool = False,
                 c1: float = 2, c2: float = 2, c3: float = None,
                 swarm_size: int = 15, number_of_clans: int = 4, current_iteration: int = 0):
        self.clans = [ClassicSwarm(fitness_function, convex_boundaries, adaptive=adaptive,
                                   maximum_iterations=maximum_iterations,
                                   # w=w,
                                   c1=c1, c2=c2, c3=c3, swarm_size=swarm_size, current_iteration=current_iteration)
                      for i in range(number_of_clans)]
        self.__fitness_function = fitness_function

        self.__max_iterations = maximum_iterations

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
                leaders, convex_boundaries=[],maximum_iterations=self.__max_iterations,
                current_iteration=self.clans[0].current_iteration)
            # Note: all clans have the same value in their "current_iteration" variable.
            clan_leaders_swarm.update_swarm()

        # Execute particle movement as per the classic PSO
        for swarm in self.clans:
            swarm.update_swarm()
        # Execute Clan PSO movement operation.
        update_clan_leaders(find_clan_leaders())

    def find_population_global_best_position(self):
        best_position = self.clans[0].global_best_position
        best_value = self.__fitness_function(self.clans[0].global_best_position)
        for swarm in self.clans:
            swarm_best_value = self.__fitness_function(swarm.global_best_position)
            if swarm_best_value > best_value:
                best_position = swarm.global_best_position
                best_value = swarm_best_value

        return best_position

    def calculate_swarm_distance_from_swarm_centroid(self):
        return 1/len(self.clans) *\
               sum(clan.calculate_swarm_distance_from_swarm_centroid()
                                       for clan in self.clans)



