"""
Copyright (C) 2021  Konstantinos Stavratis
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

from enum import Enum


class WallTypes(Enum):
    """
    Once one of the dimensions of a particle hit the boundary of the solution space, the behaviour of the swarm should be dictated to
    the walls' type.
    - NONE: the swarm's particles are not affected in any way.
    - ABSORBING: the absorbing wall sets the velocity in that corresponding dimension to zero.
    - ELIMINATING: the particles which have exited the domain are replaced by new, random particles.
    - INVISIBLE:  In order to reduce the calculation time and avoid affect the motions of other particles,
                    the invisible walls did not calculate the fitness values of the particles flying out of the boundary. 
    - REFLECTING: the reflecting wall changes the direction of particle velocity,
                    the particle and thus shall eventually be pulled back to the allowable solution space by the two walls.
    """
    NONE = 0
    ELIMINATING = 1
    ABSORBING = 2
    REFLECTING = 3
    INVISIBLE = 4
