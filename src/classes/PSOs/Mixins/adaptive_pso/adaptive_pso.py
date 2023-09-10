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

import warnings

import numpy as np
import numpy.typing as npt

import numbers

from .utils.evolutionary_state_classification_singleton_method import classify_evolutionary_state
from .utils.acceleration_coefficients_adaptation import determine_acceleration_coefficients
from .utils.eliticism_learning_strategy import eliticism_learning_strategy

class AdaptivePSO:
    """
    The Adaptive Particle Swarm Optimization (APSO) as described in the paper:
    "Zhan, Zhi-Hui & Zhang, Jun & Li, Yun & Chung, Henry. (2010).
    Adaptive Particle Swarm Optimization.
    Systems, Man, and Cybernetics, Part B: Cybernetics, IEEE Transactions on. 39. 1362 - 1381. 10.1109/TSMCB.2009.2015956.
    "

    As a short overview of the paper, the adaptivity of the swarm is done as follows:
    STEP 1: Compute the evolutionary factor of the swarm.
    STEP 2: Classify the swarm into one of the four states: "Exploration", "Exploitation", "Convergence", "Jump-Out".
            and adapt the learning factors according the class. 
    STEP 2b: Introduce a small energy (perturbation) into the system in case it is in the "Convergence" state.
    STEP 3: Update the inertia weight ω as a function of the evolutionary factor # Math: \omega = \frac{1}{1 + 1.5\exp{(-2.6 \cdot f_{evol})}} 

    IMPORTANT: This mixin requires to be incorporated into a set of classes which contains
    "self._current_iteration" and "self._maximum_iterations" variables.
    One such example is the "StandardPSO" class (mixin).
    
    Attributes
    ----------
    `c_min` : `float`
        Lower bound below which the c1 and c2 learning factors of the swarm are clampled
        The value the paper proposes is 1.5
    
    `c_max` : `float`
        Upper bound after which the c1 and c2 learning factors of the swarm are clamped.
        The value the paper proposes is 2.5

    `_evolutionary_state` : `EvolutionaryStates`
    """
    def __init__(self, c_min : float, c_max : float, **kwargs : dict):
        super().__init__(**kwargs)

        self.__c_min, self.__c_max = c_min, c_max
        self._evolutionary_state = None # Attribute declaration

    def _update_weights_and_acceleration_coefficients(self) -> None:
        super()._update_weights_and_acceleration_coefficients() # Trigger any updates done by other PSO versions

        # The evolutionary factor, f_evol, is required to update both the inertia weight --with the use of a closed-form formula--
        # and the acceleration coefficients c1 & c2.
        f_evol = self.__estimate_evolutionary_state()

        # Adapt inertia weight
        self.w = self._adapt_inertia_factor(f_evol)

        # Adapt acceleration_coefficients
        self._evolutionary_state = self.__classify_evolutionary_state(f_evol)
        self.c1, self.c2 = determine_acceleration_coefficients(self._evolutionary_state, self.c1, self.c2, self.__c_min, self.__c_max)

        # Apply eliticism learning strategy (ELS)
        index_of_gbest = np.where(np.all(self.pbest_positions == self.gbest_position, axis=1))[0] # Search particle which found gbest at some point.
        # There is the possibility that more than one particles have the same `pbest_positions`.
        # Such an example could be when more than one particles have reached the (local) optimum.
        if index_of_gbest.size > 1:
            index_of_gbest = np.random.default_rng().choice(index_of_gbest)
        index_of_gbest = index_of_gbest.item()

        self.swarm_positions = eliticism_learning_strategy(
            self._evolutionary_state, self.swarm_positions,
            index_of_gbest, self.gbest_position,
            self._objective_function, self._domain_boundaries,
            self._current_iteration, self._max_iterations
        )

    def __classify_evolutionary_state(self, f_evol : float):
        # Sanity check
        if (isinstance(f_evol, numbers.Real)):
            return classify_evolutionary_state(f_evol)
        else:
            raise TypeError(f'Expected type of parameter evolutionary factor f_evol to be rational number {type(numbers.Real)}.\n'\
            f'{type(f_evol)} was provided instead.')
        
    def __estimate_evolutionary_state(self):

        return self.__compute_evolutionary_factor(self.__average_between_each_particle_to_all_others())
    
    def _adapt_inertia_factor(self, f_evol: float):
        return 1 / (1 + 1.5 * np.exp(-2.6 * f_evol))  # ∈[0.4, 0.9]  ∀f_evol ∈[0,1]
    

    def __compute_evolutionary_factor(self, d : npt.NDArray[np.float_]) -> float:
        """
        Parameters
        ----------
        d : np.array
            - shape = (n, ), where n is the number of rows of "self.swarm_positions",
            which in turn encodes the number of particles in the swarm.

            - d[i] := The average (euclidean) distance of particle i with all other particles j: # Math: d_i = \frac{1}{N-1} \sum_{i=1}^{N} \| \mathbf{x}_i - \mathbf{x}_j \| 

        Returns
        -------
        : float
            The evolutionary factor: # Math: f_{evol} = \frac{d_g - d_{min}}{d_{max} - d_{min}}
        """

        # Compute the index of the globally best particle
        g = np.where(np.all(self.pbest_positions == self.gbest_position, axis=1))[0]
        # There is the possibility that more than one particles have the same `pbest_positions`.
        # Such an example could be when more than one particles have reached the (local) optimum.
        if g.size > 1:
            g = np.random.default_rng().choice(g)
        g = g.item()

        
        # NOTE: Explanation as to why the above command makes sense.
        # Quoting the paper "Denote di of the globally best particle dg".
        # A question that may arise is "how is the globally best particle known at all times?"
        # There are only two cases:
        #   a) The paper refers to the particle which at some point had the global position.
        #   b) The paper refers to the particle which has the best objective value at the current iteration.
        #   There are cases where a) and b) are not the same particle.
        #   This may happen when the particles "shoot over" better solutions, "landing" on only worse than the currently global best.
        # This implementation assumes case a), because it does not require passing information inside the swarm about the objective values.
        # In case such information is stored, the current implementation may be changed to that.
        # The paper itself, however, does NOT make it clear which one of the two it is referring to.

        d_g, d_min, d_max = d[g], np.min(d), np.max(d)

        # NOTE: The evolutionary factor is bound in the values [0,1].
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                evolutionary_factor : float = (d_g - d_min) / (d_max - d_min)
            except RuntimeWarning: evolutionary_factor = 0.0
            # The only way that a runtime warning will arise in this case is a division by zero.
            # This would imply that d_min = d_max.
            # This is only possible when all particles are present on the same point in space.
            # Consequently, we may confidently claim that the swarm is in a state of (ultimate) convergence.
            # Therefore, the lowest possible evolutionary factor possible is returned.
        return evolutionary_factor




    
    def __average_between_each_particle_to_all_others(self):
        """
        Returns
        -------
        : np.array
            shape = (n, ), where n is the number of rows of the "self.swarm_positions" attribute
            (which in turn encodes the number of particles of the swarm).

            output[i] := average distance (2-norm) of particle i w.r.t. all other particles of the swarm.
        """
        # particle_difference_vectors[particle i][particle j][dimension d] := difference vector between particles i and j (i-j) 
        particle_difference_vectors = np.apply_along_axis(lambda row: row - self.swarm_positions, arr=self.swarm_positions, axis=1)
        two_norm_of_vector_i_j = np.linalg.norm(particle_difference_vectors, ord=2, axis=-1)
        # Math: d_i = \frac{1}{N-1} \sum_{i=1}^{N} \| \mathbf{x}_i - \mathbf{x}_j \|
        average_distances_between_i_j = np.sum(two_norm_of_vector_i_j, axis=1) / two_norm_of_vector_i_j.shape[0]

        return average_distances_between_i_j