from .enums.global_local_coefficient_types import GlobalLocalCoefficientTypes as glct
from .enums.control_factor_types import ControlFactorTypes as cft

c3_start, c3_end = 2.0, 0.0  # We'll have to see whether this line of code is executed according to the import command. e.g. import only class, is this being executed?

class EnhancedInformationSharingPSO(object):

    def __init__(self,
                has_eis: bool = False, global_local_coefficient_method: glct = glct.NONE, c3: float = None,
                control_factor_method: cft = cft.NONE, c3_k: float = None):

        self._has_eis = has_eis
        self._global_local_coefficient_method = global_local_coefficient_method
        self._control_factor_method = control_factor_method
        self._c3_k = c3_k

        # Sanity checks for the provided input.

        # c3 is not a number <=> c3_k is not a number
        if self._global_local_coefficient_method == glct.NONE:
            self.c3, self.c3_k = None, None


        elif self._global_local_coefficient_method == glct.CONSTANT:
            if isinstance(c3, float):
                self.c3 = c3
            else:
                raise TypeError(f"Global-local coefficient c3 that was given is not a real number {type(float)}.\n\
                Instead, {type(c3)} was provided.")

        elif self._global_local_coefficient_method == glct.LINEAR:
            self.c3 = c3  # Note that in reality, c3 is calculated dynamically at each iteration.

        elif self._global_local_coefficient_method == glct.ADAPTIVE:
            if isinstance(c3, float):
                self.c3 = c3
            else:
                raise TypeError(f"Global-local coefficient c3 that was given is not a real number {type(float)}.\n\
                Instead {type(c3)} was provided.")
            self.c3_k = None  # c3 is dependent on c3_k only in the case of a linear c3.

        else:
            ValueError("A not defined Enhanced Information Sharing (EIS) methodology has been given.")

        # c3 is a number <=> c3_k is a number
        if self._global_local_coefficient_method == glct.CONSTANT or \
                self._global_local_coefficient_method == glct.LINEAR or \
                self._global_local_coefficient_method == glct.ADAPTIVE:
            if isinstance(c3_k, float):
                self.c3_k = c3_k
            else:
                raise TypeError(f"The value assigned to the control factor c3_k is not a floating-point number {type(float)}.\n\
                Instead, {type(c3_k)} was provided.")

        if self._control_factor_method == cft.LINEAR:
            self.__c3_k_start = c3_k


    def _update_global_local_coefficient(self):
        # Changing global-local coefficient c_3
        if self._global_local_coefficient_method == glct.NONE or \
                self._global_local_coefficient_method == glct.CONSTANT:
            pass

        # Global optimization capability is strong when c_3 is linearly decreasing (c3_k > 0) according to the article.
        elif self._global_local_coefficient_method == glct.LINEAR:
            self.c3 = self.c3_k * (
                        c3_start - (c3_start - c3_end) / self.__max_iterations * self.current_iteration)

        elif self._global_local_coefficient_method == glct.ADAPTIVE:
            # When local exploration is encouraged, the coefficient contributes a vector starting
            # from the swarm's global best pointing towards the particle's local best.
            if self.__evolutionary_state == EvolutionaryStates.EXPLORATION or \
                    self.__evolutionary_state == EvolutionaryStates.EXPLOITATION:
                self.c3 = -(abs(self.c3))
            # When global exploration is encouraged, the coefficient contributes a vector starting
            # from the particle's best pointing towards the swarm's global best.
            elif self.__evolutionary_state == EvolutionaryStates.CONVERGENCE or \
                    self.__evolutionary_state == EvolutionaryStates.JUMP_OUT:
                self.c3 = abs(self.c3)

    
    def _update_global_local_control_factor(self):
        if self._control_factor_method == cft.NONE or \
                self._control_factor_method == cft.CONSTANT:
            pass

        # Changing global-local coefficient control factor k.
        elif self._control_factor_method == cft.LINEAR:
            self.c3_k = self.__c3_k_start - (
                        self.__c3_k_start - 0) / self.__max_iterations * self.current_iteration

        elif self._control_factor_method == cft.ADAPTIVE:
            # When local exploration is encouraged, the coefficient contributes a vector starting
            # from the swarm's global best pointing towards the particle's local best.
            if self.__evolutionary_state == EvolutionaryStates.EXPLORATION or \
                self.__evolutionary_state == EvolutionaryStates.EXPLOITATION:
                self.c3_k = -(abs(self.c3_k))
            # When global exploration is encouraged, the coefficient contributes a vector starting
            # from the particle's best pointing towards the swarm's global best.
            elif self.__evolutionary_state == EvolutionaryStates.CONVERGENCE or \
                    self.__evolutionary_state == EvolutionaryStates.JUMP_OUT:
                self.c3_k = abs(self.c3_k)


