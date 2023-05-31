import numpy as np

class StandardPSO():
    def __init__(self, w_min : float, w_max : float, max_iterations : int):
        self._w_min, self._w_max = w_min, w_max
        self.__max_iterations = max_iterations
        self.__current_iteration = 1
    
    def update_weights_and_coefficients(self):
        """
        In standard PSO, the inertia weight linearly decreases from the maximum value of ω to its minimum value.
        # Math: \omega = \omega_{max} - (\omega_{max} - \omega_{min}) \frac{t}{t_{max}}
        """
        self.w = self.__w_max - (self._w_max - self._w_min) * (self.__current_iteration / self.__max_iterations)
        self.__current_iteration += 1
        # TODO: Check if the increment command has to be added to a different function (e.g. "step_velocities" instead).
        # This could happen in cases where the programmer accidentally calls the "update_weights" multiple times, 
        # without going to the next iteration of the algorithm.
        # With the current mixins, no such need is required (i.e. re-update weights in a single iteration).