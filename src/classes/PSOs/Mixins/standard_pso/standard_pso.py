"""
Copyright (C) 2024  Konstantinos Stavratis
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

import warnings

class StandardPSO:
    '''
    The `StandardPSO` mixin incorporates a time-varying inertia weight.

    In particular, this mixin introduced the linearly decreasing inertia weight
    as first proposed in the paper "Empirical study of particle swarm optimization"
    by Shi and Eberhart (https://doi.org/10.1109/CEC.1999.785511).

    In it, the following scheme to linearly decrease the inertia weight ω is as follows:
    `ω = ω_max - (ω_max - ω_min) * (t / t_max)`

    Attributes
    ----------
    `max_iterations` : `int`
        The maximum number of steps the PSO algorithm is expected to take (t_max).

    `current_iteration` : `int`
        The number of steps the PSO algorithm has taken so far (t).

    `w_min` : `float`
        The minimum value that the inertia weight ω
        will reach at the final PSO algorithm step, ω_min.
        This value is held when `current_iteration == max_iterations`.

    `w_max` : `float`
        The maximum value of the inertia ω, ω_max
        This value is held when `current_iteration = 0`. 
    '''

    @property
    def current_iteration(self) -> int:
        return self.__current_iteration
    
    # The `__current_iteration` variable should *not* be able to be
    # manipulated outside of this class.
    # Leaving the code in case future development proves otherwise. 
     
    # @current_iteration.setter
    # def current_iteration(self, value: int):
    #     if value < 0:
    #         raise ValueError('The value should represent an iteration number,\
    #                          which by definition is a natural number.'
    #                          + f'However, {value} < 0')

    #     if value > self.__max_iterations:
    #         raise ValueError(f'The value provided exceeds the expected number of maximum iterations:\
    #                          {value} > {self.__max_iterations}')
        
    #     self.__current_iteration = value

    @property
    def max_iterations(self) -> int:
        return self.__max_iterations
    
    @property
    def w_min(self):
        return self.__w_min
    
    @property
    def w_max(self):
        return self.__w_max

    def __init__(self, w_min : float, w_max : float, max_iterations : int, current_iteration : int = 0, **kwargs : dict):

        # ==================== Input parameter checks START ====================
        # ITERATIONS CHECKS
        if max_iterations < 1:
            raise ValueError('The PSO algorithm should execute at least one step. Yet\n'
                             f'`max_iterations` = {max_iterations}')

        if current_iteration < 0:
            raise ValueError('Taken steps cannot be less than "no steps" (a.k.a. 0). Yet\n'
                             f'`current_iteration` = {current_iteration}.')
        
        if current_iteration > max_iterations:
            raise ValueError('Taken steps cannot be greater than the maximum expected steps that the PSO will required. Yet\n'
                             f'{current_iteration} = current_iteration > max_iterations = {max_iterations}')
        

        # INERTIA WEIGHT ω CHECKS
        if w_min < 0:
            raise ValueError('Inertia weight ω represents a clipping factor. Therefore all its values should exceed zero.'\
                             f'Yet the provided value {w_min} < 0.')
        if w_max < 0:
            raise ValueError('Inertia weight ω represents a clipping factor. Therefore all its values should exceed zero.'\
                             f'Yet the provided value {w_min} < 0.')
        
        if w_min > w_max:
            warnings.warn(message=f'The provided {w_min} = ω_min >  ω_max = {w_max}, '
                          'which by definition should not be the case (max < min ?).\n'
                          f'Flipping values to ω_min = {w_max} and ω_max = {w_min} instead.', category=UserWarning)
            w_min, w_max = w_max, w_min

        if w_min == w_max:
            warnings.warn(message=f'ω_min = ω_max = {w_min}.'
                          'No time-dependent decrease of the inertia weight will take place.')
        # ==================== Input parameter checks FINISH ====================


        self.__w_min, self.__w_max = w_min, w_max
        self.__max_iterations = max_iterations
        self.__current_iteration = current_iteration

        super().__init__(**kwargs) # Call constructor of the exact immediate ancestor in the class hierarchy.
    
    def _update_weights_and_acceleration_coefficients(self) -> None:
        """
        In standard PSO, the inertia weight linearly decreases from the maximum value of ω to its minimum value.
        # Math: \omega = \omega_{max} - (\omega_{max} - \omega_{min}) \frac{t}{t_{max}}
        """
        super()._update_weights_and_acceleration_coefficients() # Trigger any updates done by other PSO versions

        self.w = self.w_max - (self.w_max - self.w_min) * (self.current_iteration / self.max_iterations)
        self.__current_iteration += 1
        # TODO: Check if the increment command has to be added to a different function (e.g. "_step_velocities" instead).
        # This could happen in cases where the programmer accidentally calls the "update_weights" multiple times, 
        # without going to the next iteration of the algorithm.
        # With the current mixins, no such need is required (i.e. re-update weights in a single iteration).

        if self.__current_iteration > self.__max_iterations:
            raise OverflowError('PSO steps have exceeded the maximum iterations provided during initialization.'\
                                f'maximum iterations = {self.max_iterations}, current iteration = {self.current_iteration}')