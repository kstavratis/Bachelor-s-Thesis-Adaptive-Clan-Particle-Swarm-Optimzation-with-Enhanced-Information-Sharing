"""
This file handles extracting attributes of interest from a swarm configuration
and "packaging" them into `pandas.DataFrame` structures.
Currently implemented attributes of interest are:
- the positions of all particles
- the global best position of the swarm

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

import pandas as pd

from typing import Type

from src.classes.PSOs.pso_backbone import PSOBackbone
from src.classes.PSOs.clan_pso import ClanPSO

def swarm_positions_extractor(swarm : Type[ClanPSO or PSOBackbone]) -> pd.DataFrame:
    """
    Parameters
    ----------
    `swarm` : `Type[ClanPSO or PSOBackbone]`
        A swarm instance.
        Currently, the swarms supported are `PSOBackbone` and `ClanPSO` classes.

    Returns
    -------
    `pd.DataFrame`
        A dataframe containing the positions of the swarm at the current configuration in the following format.\n
        ```
        -----------------------------------------------------------------\n
                |               | dimension0    |   ... |   dimensionD-1|\n
        clanID  |   particle0   |               |   ... |   ...         |\n
                |   ...         |               |   ... |   ...         |\n
                |   particleP-1 |               |   ... |   ...         |\n
        -----------------------------------------------------------------\n
        ```
        
    Errors
    ------
    `NotImplementedError` : An error is raised in cases where a non-expected swarm instance is provided as input.
    Currently, the instances of PSO supported are: `PSOBackbone`, `ClanPSO`.
    """
    
    if isinstance(swarm, ClanPSO):
        return _clan_pso_positions_writer(swarm)
    elif isinstance(swarm, PSOBackbone):
        return _backbone_pso_positions_writer(swarm)
    else:
        raise NotImplementedError('The current implementation of the software can only handle "PSOBackbone" and "ClanPSO" variations.')


    
def _backbone_pso_positions_writer(swarm : Type[PSOBackbone]) -> pd.DataFrame:
    
    particles_index = pd.Index([i for i in range(swarm.nr_particles)], dtype=int, name='particle')
    dimensions_index = pd.Index([i for i in range(swarm.nr_dimensions)], dtype=int, name='dimension')
    positions_df = pd.DataFrame(swarm.swarm_positions, index=particles_index, columns=dimensions_index)
    
    # (Manually) providing the (obvious) information that a single swarm is a single clan.
    positions_df.index = pd.MultiIndex.from_product([[0], positions_df.index], names=['clan', 'particle'])

    return positions_df

def _clan_pso_positions_writer(swarm : Type[ClanPSO]) -> pd.DataFrame:

    list_of_clan_df = []
    for clan_index, clan in enumerate(swarm.clans):

        clan_df = pd.DataFrame(
            clan.swarm_positions,
            index=pd.MultiIndex.from_product([[clan_index], [i for i in range(clan.nr_particles)]], names=['clan', 'particle']),
            columns= pd.Index([i for i in range(clan.nr_dimensions)], dtype=int, name='dimension')
        )
        
        list_of_clan_df.append(clan_df)

    return pd.concat(list_of_clan_df, axis=0)


def gbest_position_extractor(swarm : Type[ClanPSO or PSOBackbone]) -> pd.DataFrame:

    gb = swarm.gbest_position

    gbest_df = pd.DataFrame(
        gb.reshape(1, -1), # By default, 1D arrays are made a column in pd.DataFrame. Therefore, reshaping.
        columns=[f'dimension{i+1}' for i in range(gb.size)],
    )

    return gbest_df
