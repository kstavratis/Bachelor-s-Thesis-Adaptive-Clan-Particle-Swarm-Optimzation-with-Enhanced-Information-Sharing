"""
Copyright (C) 2023  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""


import numpy as np

class PSOBackbone:
    """
    The "PSOBackone" class encompasses all functionalities essential for any Particle Swarm Optimization (PSO) algorithm.

    Attributes
    ----------
    __nr_particles : int
        A positive integer which stores the number of particles which composite the swarm
    
    __nr_dimensions : int
        The dimensionality (i.e. number of degrees of freedom) of the optimization problem that the PSO algorithm will solve.

    swarm_positions : np.array
        size == (__nr_particles, __nr_dimensions)\n
        A 2D array whose cells represent the positions of the swarm's particles.
        The rows index the particle ID, while the column the DOF id.

    swarm_velocities : np.array
        size == (__nr_particles, __nr_dimensions)\n
        A 2D array whose cells represent the velocities of the swarm's particles.
        The rows index the particle ID, while the column the DOF id.

    __objective_function : function
        A (class <function>) pointer to the problem landscape, expressed in a mathematical formula, the PSO is tasked with solving.
        The function should be conditioned in such a way such that it produces (__nr_particles,) np.array outputs
        from a (__nr_particles, __nr_dimensions) 2D np.array. i.e.
        function(np.array(n,m)) -> np.array(n)

    w : float
        The w (ω) value is known as the "inertia weight". w # Math: \omega \in [0, 1]
        The w weight dictates how much of the previously attained velocity will be used in the next iteration of the algorithm
        (0 := not at all, 1 := fully).
        It is initialized to one (1).

    c1 : float
        The c1 value is known as "cognitive weight" or "greedy learning factor"
        Its value (in conjunction with c2) determines the behaviour of each particle of the swarm.
        As a rule of thumb, the higher the ratio c1/c2 is, the more local exploitation is encouraged.

    c2 : float
        The c2 value is known as "social weight" or "social learning factor"
        Its value (in conjunction with c1) determines the behaviour of each particle of the swarm.
        As a rule of the thumb, the higher the ration c2/c1 is, the more global exploration is encouraged.
    
        

    Although it can be instantiated on its own, it is suggested that additional schemes are built on top of this class,
    such that desirable properties (exploration, convergence, stability) are achieved.
    The recommended way of building additional PSO schemes is by implementing them as Python "Mixins".
    """

    def __init__(self, nr_particles : int, nr_dimensions : int,
                 objective_function, domain_low_boundaries : np.array, domain_high_boundaries : np.array): 
        super().__init__()

        # ==================== INITIALIZE ATTRIBUTES FROM INPUT PARAMETERS START ====================
        self.__nr_particles = nr_particles
        self.__nr_dimensions = nr_dimensions
        # Initialize a random swarm of particles, represented by a 2D numpy array of dimensions [nr_particles, nr_dimensions]
        # Currently, the initialization is done using a uniform distribution.
        self.swarm_positions = np.random.default_rng().uniform(low=domain_low_boundaries, high=domain_high_boundaries)
        # Currently, the particles will start with zero velocity.
        self.swarm_velocities = np.zeros(shape=(nr_particles, nr_dimensions))

        self.__objective_function = objective_function

        # ==================== INITIALIZE ATTRIBUTES FROM INPUT PARAMETERS FINISH ====================






        # ==================== INITIALIZE ATTRIBUTES FROM INTERNAL FUNCTIONS START ====================

        # Initialize "pbest" and "gbest".

        # Algorithm for updating the swarm.
        # STEP 1: Compute the f(x) --a.k.a. "objective values"-- for each particle (row).
        # Branch a: global best
        # Branch b: personal best

        # STEP 2a: Find the particle with the lowest value.
        # STEP 3a: Update the global best value and position
        #           in case that f(x) is lower than the one currently memorized by the swarm "gbest_value".
        
        # STEP 2b: For each particle, update the personal best value and position
        #           in case that f(x) is lower than the currently memorized objective values "pbest_value"

        # NOTE: During the initialization step (i.e. only inside the "__init__" function), no comparison checks need to be applied.
        # This is because no f(x) has been computed yet.
        # Therefore, the initial positioning is the currently best presented to the swarm.
        
        # STEP 1
        objective_values = self.__objective_function(self.swarm_positions)

        # STEP 2a
        best_objective_value_index = np.argmin(objective_values)

        # STEP 3a
        self.gbest_value = objective_values[best_objective_value_index]
        self.gbest_position = np.copy(self.swarm_positions[best_objective_value_index])
        #! Make sure to copy by value. Simple indexing copies by reference (view).
        #! For more information, read https://numpy.org/doc/stable/user/basics.copies.html


        # STEP 2b
        self.pbest_values = np.copy(objective_values)
        self.pbest_positions = np.copy(self.swarm_positions)
        #! Make sure to copy by value. Simple indexing copies by reference (view).
        #! For more information, read https://numpy.org/doc/stable/user/basics.copies.html
        # ==================== INITIALIZE ATTRIBUTES FROM INTERNAL FUNCTIONS START ====================




        # ==================== INITIALIZE ATTRIBUTES ARBITRARILY START ====================
        self.w = 1.0
        self.c1, self.c2 = 2.0, 2.0
        # ==================== INITIALIZE ATTRIBUTES FROM ARBITRARILY FINISH ====================
    
    def step(self) -> None:
        """
        Apply the velocity and position updates of the particles.
        # Math: \mathbf{v}^{(t+1)} = f(\mathbf{x^t}, \mathbf{p^t}, \mathbf{g^t}, ...)
        # Math: \mathbf{x}^{(t+1)} = \mathbf{x}^t + \mathbf{v}^{(t+1)}
        """
        self.update_weights_and_acceleration_coefficients()
        self.step_velocities()
        # Translate the swarm positions according to the updated velocities.
        self.swarm_positions += self.swarm_velocities

    def step_velocities(self) -> None:
        random_generator = np.random.default_rng()
        # NOTE: Because we are multiplying with random values, whether we do matrix-wise multiplication (@)
        # or element-wise multiplication (*) should be identical.
        # Element-wise multiplication is used, because it is believed to be computationally cheaper, especially memory-wise.

        # # ==================== Matrix-wise multiplication approach START ====================
        # # Reshaping the swarm matrix into a single (column) vector to utilize vectorization of numpy.
        # # Both representations (3D matrix x 2D matrix, larger 2D matrix, large 1D vector) are equivalent.
        # R1 = np.diag(random_generator.uniform(size=self.__nr_particles * self.__nr_dimensions))
        # R2 = np.diag(random_generator.uniform(size=self.__nr_particles * self.__nr_dimensions))
        # cognitive_velocities = (R1 @ (self.pbest_positions - self.swarm_positions).flatten()).reshape(self.__nr_particles, self.__nr_dimensions)
        # social_velocities = (R2 @ (self.gbest_position - self.swarm_positions).flatten()).reshape(self.__nr_particles, self.__nr_dimensions)
        # # ==================== Matrix-wise multiplication approach FINISH ====================

        # ==================== Element-wise multiplication approach START ====================
        R1 = random_generator.uniform(size=(self.__nr_particles, self.__nr_dimensions))
        R2 = random_generator.uniform(size=(self.__nr_particles, self.__nr_dimensions))
        cognitive_velocities = (R1 * (self.pbest_positions - self.swarm_positions))
        social_velocities = (R2 * (self.gbest_position - self.swarm_positions))
        # ==================== Element-wise multiplication approach FINISH ====================

        self.swarm_velocities = self.w * self.swarm_velocities + self.c1 * cognitive_velocities + self.c2 * social_velocities

    def update_pbest(self, candidate_positions : np.array, candidate_objective_values : np.array) -> None:
        # For each particle,
        # in case that f(x) is lower than the currently memorized objective values "pbest_value"
        # update the personal best value and position

        # A boolean mask whose "True" values indicate which values will be replaced. NumPy allows boolean indexing.
        update_mask = candidate_objective_values < self.pbest_values

        self.pbest_values[update_mask] = candidate_objective_values[update_mask]
        self.pbest_positions[update_mask] = candidate_positions[update_mask]

    def update_gbest(self, candidate_positions : np.array, candidate_objective_values : np.array) -> None:
        # STEP 1: Find the particle with the lowest value.
        best_objective_value_index = np.argmin(candidate_objective_values)

        # STEP 2: Update the global best value and position
        #           in case that f(x) is lower than the one currently memorized by the swarm "gbest_value".
        if candidate_objective_values[best_objective_value_index] < self.gbest_value:
            self.gbest_value = candidate_objective_values[best_objective_value_index]
            self.gbest_position = candidate_positions[best_objective_value_index]
        #! During testing, beware to check for copy by value or by reference! This can be done by checking the "base" attribute of np arrays.
        #! Make sure to copy by value. Simple indexing copies by reference (view).
        #! For more information, read https://numpy.org/doc/stable/user/basics.copies.html

    
    def update_weights_and_acceleration_coefficients(self):
        # The definition of this function acts as a placeholder for any other PSO mixin which behaves in a particular manner.
        # For example, in the standard PSO, the "update_weights" function could refer to the linear decrease of the inertia weight ω.
        self.super().update_weights()



    def get_swarm_centroid(self):
        return np.mean(self.swarm, axis=0)