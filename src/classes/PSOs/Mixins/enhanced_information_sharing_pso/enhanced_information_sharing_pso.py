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

from .enums.global_local_coefficient_types import GlobalLocalCoefficientTypes as glct
from .enums.control_factor_types import ControlFactorTypes as cft
from ..adaptive_pso.enums.evolutionary_states import EvolutionaryStates

class EnhancedInformationSharingPSO:
    """
    The Enhanced Information Sharing Particle Swarm Optimization (EIS PSO) introduces a new velocity update rule
    described in the paper
    "Improved Particle Swarm Optimization Algorithm Based on Last-Eliminated Principle and Enhanced Information Sharing"
    (for details, read https://doi.org/10.1155/2018/5025672).\n
    In addition, to account for other possible mixin combinations, more functionalities may be implemented.
    The current implementation is compatible with Standard PSO and Adaptive PSO.

    IMPORTANT: This mixin requires to be incorporated into a set of classes which contains
    "self._current_iteration" and "self._maximum_iterations" variables.
    One such example is the "StandardPSO" class (mixin).
    """

    c3_start, c3_end = 2.0, 0.0

    def __init__(self,
                 global_local_coefficient_method: glct or str = glct.LINEAR, c3: float = None,
                control_factor_method: cft or str = cft.CONSTANT, c3_k: float = None,
                **kwargs : dict):
        """
        Parameters
        ----------
        c3 : float

        c3_k : float
            Control factor of the c3 acceleration coefficient.
            It is multiplied unto c3, with the intended purpose of alleviating the effect of c3 on the velocities.

        global_local_coefficient_method : str
            It may be receive one of the following values: `'constant'`, `'linear'` or `'adaptive'`

        control_factor_method : str
            It may be receive one of the following values: `'constant'`, `'linear'` or `'adaptive'`

        kwargs : dict
            Keyword arguments which shall be used for initializers of superclasses.

        NOTE: It is advised to have both `global_local_coefficient_method` and `control_factor_method` to adaptive behaviour
        when wishing to observe an adaptive behaviour from the swarm. Anything else may lead to unexpected behaviour.
        """
        
        super().__init__(**kwargs)

        # String-formatted input may be more intuitive to users, rather than having to provide the actual enum themselves.
        # ==================== Handling cases where the input was provided in string (`str`) format START ====================
        if type(global_local_coefficient_method) == str:
            global_local_coefficient_method = global_local_coefficient_method.upper() # Make all the letters capitals.
            global_local_coefficient_method = getattr(glct, global_local_coefficient_method)

        if type(control_factor_method) == str:
            control_factor_method = control_factor_method.upper() # Make all the letters capitals.
            control_factor_method = getattr(cft, control_factor_method)
        # ==================== Handling cases where the input was provided in string (`str`) format FINISH ====================
        
        self._global_local_coefficient_method = global_local_coefficient_method
        self.__control_factor_method = control_factor_method
        self._c3_k = c3_k

        #################### Sanity checks and assignments for the provided input START ####################

        if self._global_local_coefficient_method == glct.CONSTANT:
            if isinstance(c3, float):
                self.c3 = c3
            else:
                raise TypeError(f'Global-local coefficient c3 that was given is not a real number {type(float)}.\n'\
                f'Instead, {type(c3)} was provided.')

        elif self._global_local_coefficient_method == glct.LINEAR:
            self.c3 = c3  # Note that in reality, c3 is calculated dynamically at each iteration.

        elif self._global_local_coefficient_method == glct.ADAPTIVE:
            if isinstance(c3, float):
                self.c3 = c3
            else:
                raise TypeError(f"Global-local coefficient c3 that was given is not a real number {type(float)}.\n\
                Instead {type(c3)} was provided.")
            self._c3_k = None  # c3 is dependent on c3_k only in the case of a linear c3.

        else:
            raise NotImplementedError("A not defined Enhanced Information Sharing (EIS) methodology has been given.")

        # c3 is a number <=> c3_k is a number
        if self._global_local_coefficient_method == glct.CONSTANT or \
                self._global_local_coefficient_method == glct.LINEAR or \
                self._global_local_coefficient_method == glct.ADAPTIVE:
            if isinstance(c3_k, float):
                self._c3_k = c3_k
            else:
                raise TypeError(f"The value assigned to the control factor c3_k is not a floating-point number {type(float)}.\n\
                Instead, {type(c3_k)} was provided.")

        if self.__control_factor_method == cft.LINEAR:
            self.__c3_k_start = c3_k

        #################### Sanity checks and assignments for the provided input FINISH ####################

    def _step_velocities(self):
        """
        The update rule to be applied depends on whether the control factor is adaptive
        (`control_factor_method == ControlFactorTypes.ADAPTIVE`)
        - If the control factor method is not adaptive => # Math: \mathbf{v}^{(t+1)} =  \omega \mathbf{v}^{t} + c_1 R_1 (\mathbf{pbest}^{t} - \mathbf{x}^{t}) + c_2 R_2 (\mathbf{gbest}^{t} - \mathbf{x}^{t}) + c_3 R_3 |\mathbf{gbest}^{t} - \mathbf{pbest}^{t}|^{abs}
        - If the control factor method is adaptive => # Math: \mathbf{v}^{(t+1)} =  \omega \mathbf{v}^{t} + c_1 R_1 (\mathbf{pbest}^{t} - \mathbf{x}^{t}) + c_2 R_2 (\mathbf{gbest}^{t} - \mathbf{x}^{t}) + c_3 R_3 (\mathbf{gbest}^{t} - \mathbf{pbest}^{t})

        Note: The actual direction at which (gbest - pbest) is pointing is dependent on the sign of c3,
        which in turn is dependent on the mode of the `global-local coefficient` input argument.\n
        It is advised to have both `global_local_coefficient_method` and `control_factor_method` to adaptive behaviour
        when wishing to observe an adaptive behaviour from the swarm. Anything else may lead to unexpected behaviour.
        
        """
        random_generator = np.random.default_rng()
        # NOTE: Because we are multiplying with random values, whether we do matrix-wise multiplication (@)
        # or element-wise multiplication (*) should be identical.
        # Element-wise multiplication is used, because it is believed to be computationally cheaper, especially memory-wise.

        # ==================== Element-wise multiplication approach START ====================
        R1 = random_generator.uniform(size=self.swarm_positions.shape)
        R2 = random_generator.uniform(size=self.swarm_positions.shape)
        R3 = random_generator.uniform(size=self.swarm_positions.shape)
        cognitive_velocities = self.c1 * R1 * (self.pbest_positions - self.swarm_positions)
        social_velocities = self.c2 * R2 * (self.gbest_position - self.swarm_positions)

        # Variable declaration
        local_global_velocities = None 
        if self.__control_factor_method != cft.ADAPTIVE: # Follow paper's approach
            local_global_velocities = self.c3 * R3 * np.abs(self.gbest_position - self.pbest_positions) # "gbest_position" is broadcast.
        else: # Follow my approach
            local_global_velocities = self.c3 * R3 * (self.gbest_position - self.pbest_positions) # The direction is determined by the sign of c3.
        # ==================== Element-wise multiplication approach FINISH ====================

        self.swarm_velocities = self.w * self.swarm_velocities + cognitive_velocities + social_velocities + local_global_velocities
        

    def _update_weights_and_acceleration_coefficients(self):
        super()._update_weights_and_acceleration_coefficients() # Trigger any updates done by other PSO versions.

        # NOTE: Order matters! The control factor must be updated first before the c3 coefficient is computed.
        self.__update_global_local_control_factor()
        self.__update_global_local_coefficient()


    def __update_global_local_coefficient(self):
        # Changing global-local coefficient c3
        if self._global_local_coefficient_method == glct.CONSTANT:
            pass

        # Global optimization capability is strong when c3 is linearly decreasing (c3_k > 0) according to the article.
        elif self._global_local_coefficient_method == glct.LINEAR:
            # c3 = c3_k * (c3_s - (c3_s - c3_e) * (t/t_max))
            self.c3 = self._c3_k * (
                        EnhancedInformationSharingPSO.c3_start\
                            - (EnhancedInformationSharingPSO.c3_start - EnhancedInformationSharingPSO.c3_end) *\
                                (self._current_iteration / self._max_iterations)
            )

        elif self._global_local_coefficient_method == glct.ADAPTIVE:
            # When local exploration is encouraged, the coefficient contributes a vector starting
            # from the swarm's global best pointing towards the particle's local best.
            if self._evolutionary_state == EvolutionaryStates.EXPLORATION or \
                    self._evolutionary_state == EvolutionaryStates.EXPLOITATION:
                self.c3 = -(abs(self.c3))
            # When global exploration is encouraged, the coefficient contributes a vector starting
            # from the particle's best pointing towards the swarm's global best.
            elif self._evolutionary_state == EvolutionaryStates.CONVERGENCE or \
                    self._evolutionary_state == EvolutionaryStates.JUMP_OUT:
                self.c3 = abs(self.c3)

    
    def __update_global_local_control_factor(self):
        if self.__control_factor_method == cft.CONSTANT:
            pass

        # Changing global-local coefficient control factor k.
        elif self.__control_factor_method == cft.LINEAR:
            self._c3_k = self.__c3_k_start - (
                        self.__c3_k_start - 0) / self._max_iterations * self._current_iteration

        elif self.__control_factor_method == cft.ADAPTIVE:
            # When local exploration is encouraged, the coefficient contributes a vector starting
            # from the swarm's global best pointing towards the particle's local best.
            if self._evolutionary_state == EvolutionaryStates.EXPLORATION or \
                self._evolutionary_state == EvolutionaryStates.EXPLOITATION:
                self._c3_k = -(abs(self._c3_k))
            # When global exploration is encouraged, the coefficient contributes a vector starting
            # from the particle's best pointing towards the swarm's global best.
            elif self._evolutionary_state == EvolutionaryStates.CONVERGENCE or \
                    self._evolutionary_state == EvolutionaryStates.JUMP_OUT:
                self._c3_k = abs(self._c3_k)
