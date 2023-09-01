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

from typing import Iterable, Type

from .pso_backbone import PSOBackbone

class ClanPSO:
    """
    The Clan Particle Swarm Optimization (Clan PSO) was first proposed by Danilo Ferreira de Carvalho and Carmelo José Albanez Bastos‐Filho,
    presented in the paper of the same name (https://doi.org/10.1108/17563780910959875).

    The main inspiration of this topology of the PSO are clans: a collection of people who are united under a shared "banner" (e.g. common ancestor).
    
    The variation that Clan PSO offers to the original PSO
    is the fact that the algorithm alternates between multiple small swarms ("clans") executing a PSO iteration each,
    and executing a PSO iteration among the best members of each clan.
    The best particles of each clan were dubbed "leaders", whiile the PSO step executed by the "leaders" was dubbed "conference of leaders".

    Attributes
    ----------
    clans : Iterable[Type[PSOBackbone]] (list, tuple)
        An iterable which contains a collection of PSO instances, which inherit from `PSOBackbone`.

    gbest_position : np.array
        Position in the search space which has been found to hold the best objective value.

    gbest_value : float
        Best (lowest) value that the swarm has (currently) found during exploration of the search space. 

    Methods
    -------
    step(conference_pso_behaviour : Iterable, conference_pso_kwargs)
    """
    
    def __init__(self, clans : Iterable[Type[PSOBackbone]]):
        self.clans : Iterable[Type[PSOBackbone]] = clans

        current_best_particles_of_clans =  [c.get_current_best_particle() for c in clans] # Find clan leaders at initialization
        clan_leaders_objective_values = np.array([c[2] for c in current_best_particles_of_clans])
        best_clan_leader_index = np.argmin(clan_leaders_objective_values)

        self.gbest_value = current_best_particles_of_clans[best_clan_leader_index][2]
        self.gbest_position = current_best_particles_of_clans[best_clan_leader_index][1]

    def step(self, conference_pso_behaviour : Iterable or Type, conference_pso_kwargs : dict = None):
        """
        Parameters
        ----------
        conference_pso_behaviour : Iterable or Class
            An iterable (tuple or list) which contains pointers to classes & mixins
            from which the behaviour of the conference shall be determined.
            It is worth stressing that the clan can follow a different behaviour (i.e. PSO variation) than each clan separately.\n
            Alternatively, the class object may be directly provided.

        conference_pso_kwargs : dict 
            A dictionary of keyword arguments (kwargs) which contains all necessary argument name-value (string : Any) pairs
            for the construction of the conference PSO instance.
            Namely, it contains enough keyword arguments to satisfy all classes encapsulated in "conference_pso_behaviour".
            
        Returns
        -------
        `None`

        An iteration of PSO determined by the arguments is enacted by using the particles with the best (lowest) value from each clan.
        The changes are enacted directly on the clans of `self.clans`.
        """

        # Firstly, execute a PSO iteration with each independent clan.
        for c in self.clans:
            c.step()



        # Proceed to prepare the conference of leaders.

        # "For each iteration, each clan performs a search and marks the particle that *had reached* the best position of the entire clan"
        # ~ https://doi.org/10.1108/17563780910959875 and https://doi.org/10.1109/SIS.2011.5952569
        # Collect best particles of each clan

        # # NOTE: Version of how I interpreted the above phrase.
        # # The particles which historically reached the best position are used.
        # temp = [c.get_reached_gbest_particle() for c in self.clans]
        # NOTE: Version of how Alkmini interpreted the above phrase
        # The particles which has the best position *for this iteration*.
        temp = [c.get_current_best_particle() for c in self.clans]

        clan_leader_indices = [t[0] for t in temp]
        clan_leaders = np.array([t[1] for t in temp])
        clan_pbest_positions = np.array([self.clans[i].pbest_positions[clan_leader_indices[i]] for i in range(len(self.clans))])
        conference_pso_kwargs['input_swarm_positions'] = clan_leaders
        conference_pso_kwargs['input_pbest_positions'] = clan_pbest_positions
        #! Hard-coding a kwarg is not good practice. Find a way to do this for all possible interesting kwargs...
        # We wish for knowledge of the current iteration to be transferred down, since it is used in most PSO variations.
        for clan in self.clans:
            if '_current_iteration' in dir(clan):
                conference_pso_kwargs['current_iteration'] = clan._current_iteration
        

        # Determine the PSO variation of the conference of leaders.
        conference_pso_class_ptr = None # Variable declaration
        if type(conference_pso_behaviour) == Type:
            conference_pso_class_ptr = conference_pso_behaviour
        elif isinstance(conference_pso_behaviour, Iterable):
            # Dynamically create a class which hosts the conference of leaders in case a class has not been provided.
            # If this code is to be changed to more dynamic format, read https://www.geeksforgeeks.org/create-classes-dynamically-in-python/
            class ConferencePSO(*conference_pso_behaviour):
                def __init__(self, kwargs : dict):
                    super().__init__(**kwargs)

            conference_pso_class_ptr = ConferencePSO
        else:
            raise ValueError('Expected type of argument "conference_pso_behaviour" to be either a pointer to class or an iterable.'
                       f'{type(conference_pso_behaviour)} was provided instead.')

        conference_instance = conference_pso_class_ptr(conference_pso_kwargs)

        # NOTE: Reminder that this PSO contains the leaders, as 'input_swarm_positions' is filled in.

        # Execute "conference of leaders" (i.e. a PSO iteration involving the best particles of each clan).
        conference_instance.step()


        # Update the locations of gbest in each clan.
        for i in range(conference_instance.swarm_positions.shape[0]):
            self.clans[i].swarm_positions[clan_leader_indices[i]] = conference_instance.swarm_positions[i] # Update movement of clan leaders.
            self.clans[i]._update_pbest_and_gbest() #* This operation is expensive, because all particles are re-evaluated, but only one particle has changed!

        # Update the the globablly(!) best value and position of the search space
        if conference_instance.gbest_value < self.gbest_value:
            self.gbest_position, self.gbest_value = conference_instance.gbest_position, conference_instance.gbest_value
        # NOTE: Why is this global?
        # 1) The "conference_instance" before the iteration --step() function call-- contains the best particles of the clans.
        #   Therefore, in there resides the single best value that has been found by the swarm (it's one of the leaders).
        # 2) After the conference of the leaders, there are two cases:
        #   a) The leaders found a new best position. Consequently, a value even better than that of 1) was found.
        #       Thus, the new best position is indeed the best position of the swarm (and with the corresponding best value).
        #   b) The leaders did not find a better position. Consequently, the value of 1) remains to be best.
        #       Therefore, the already recorded "self.gbest_position" is preserved.