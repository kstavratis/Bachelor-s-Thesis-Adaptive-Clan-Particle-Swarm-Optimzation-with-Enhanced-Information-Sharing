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

class StandardPSO:
    def __init__(self, w_min : float, w_max : float, max_iterations : int, current_iteration : int = 1, **kwargs : dict):
        self.__w_min, self.__w_max = w_min, w_max
        self._max_iterations = max_iterations
        self._current_iteration = current_iteration

        super().__init__(**kwargs) # Call constructor of the exact immediate ancestor in the class hierarchy.
    
    def _update_weights_and_acceleration_coefficients(self):
        """
        In standard PSO, the inertia weight linearly decreases from the maximum value of ω to its minimum value.
        # Math: \omega = \omega_{max} - (\omega_{max} - \omega_{min}) \frac{t}{t_{max}}
        """
        super()._update_weights_and_acceleration_coefficients() # Trigger any updates done by other PSO versions

        self.w = self.__w_max - (self.__w_max - self.__w_min) * (self._current_iteration / self._max_iterations)
        self._current_iteration += 1
        # TODO: Check if the increment command has to be added to a different function (e.g. "_step_velocities" instead).
        # This could happen in cases where the programmer accidentally calls the "update_weights" multiple times, 
        # without going to the next iteration of the algorithm.
        # With the current mixins, no such need is required (i.e. re-update weights in a single iteration).