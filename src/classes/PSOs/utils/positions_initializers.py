import numpy as np

from ..pso_backbone import PSOBackbone
from ..clan_pso import ClanPSO

from typing import Iterable, Type

def single_swarm_uniform_positions_resetter(swarm : PSOBackbone, seed : int = None):

    db = swarm._domain_boundaries
    swarm.swarm_positions = np.random.default_rng(seed).uniform(
        low=db[:, 0],
        high=db[:, 1],
        size=swarm.swarm_positions.shape
    )
    swarm.forget()

    return swarm

def clan_swarms_uniform_positions_resetter(swarm : ClanPSO, seed : int = None):

    rand_gen = np.random.default_rng(seed)
    infty = np.iinfo(np.int_).max 
    
    clans : Iterable[Type[PSOBackbone]] = []
    for c in swarm.clans:
        clans.append(single_swarm_uniform_positions_resetter(c, rand_gen.integers(infty)))
    swarm.clans = clans

    current_best_particles_of_clans =  [c.get_current_best_particle() for c in swarm.clans] # Find clan leaders at initialization
    clan_leaders_objective_values = np.array([c[2] for c in current_best_particles_of_clans])
    best_clan_leader_index = np.argmin(clan_leaders_objective_values)

    swarm.gbest_value = current_best_particles_of_clans[best_clan_leader_index][2]
    swarm.gbest_position = current_best_particles_of_clans[best_clan_leader_index][1]

    return swarm