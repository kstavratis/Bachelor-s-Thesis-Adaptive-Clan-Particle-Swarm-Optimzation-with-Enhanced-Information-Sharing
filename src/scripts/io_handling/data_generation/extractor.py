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

import pandas as pd

from typing import Type

from src.classes.PSOs.pso_backbone import PSOBackbone
from src.classes.PSOs.clan_pso import ClanPSO

def swarm_positions_extractor(swarm : Type[ClanPSO or PSOBackbone]):
    """
    Parameters
    ----------
    swarm : Type[ClanPSO or PSOBackbone]
        A swarm instance.
        Currently, the swarms supported are `PSOBackbone` and `ClanPSO` classes.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the positions of the swarm at the current configuration in the following format.\n
        -----------------------------------------------------------------\n
                |               | dimension1    |   ... |   dimensionD  |\n
        clanID  |   particle1   |               |   ... |   ...         |\n
                |   ...         |               |   ... |   ...         |\n
                |   particleP   |               |   ... |   ...         |\n
        -----------------------------------------------------------------\n

    """
    
    if isinstance(swarm, ClanPSO):
        return _clan_pso_writer(swarm)
    elif isinstance(swarm, PSOBackbone):
        return _backbone_pso_writer(swarm)
    else:
        raise NotImplementedError('The current implementation of the software can only handle "PSOBackbone" and "ClanPSO" variations.')


    
def _backbone_pso_writer(swarm : Type[PSOBackbone]):

    positions = swarm.swarm_positions
    
    positions_df = pd.DataFrame(positions,
                                index=[f'particle{i+1}' for i in range(positions.shape[0])],
                                columns=[f'dimension{i+1}' for i in range(positions.shape[1])])
    
    # (Manually) providing the (obvious) information that a single swarm is a single clan.
    positions_df.index = pd.MultiIndex.from_product([['clan1'], positions_df.index], names=['clan', 'particle'])

    return positions_df

def _clan_pso_writer(swarm : Type[ClanPSO]):

    list_of_clan_df = []
    for ci in range(len(swarm.clans)):
        positions = swarm.clans[ci].swarm_positions

        clan_df = pd.DataFrame(
            positions,
            index=pd.MultiIndex.from_product([[f'clan{ci+1}'], [f'particle{i+1}' for i in range(positions.shape[0])]], names=['clan', 'particle']),
            columns=[f'dimension{i+1}' for i in range(positions.shape[1])],
        )
        
        list_of_clan_df.append(clan_df)

    return pd.concat(list_of_clan_df, axis=0)
