"""
Copyright (C) 2023  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""


import numpy as np

class PSOBackbone():

    # The objective function of the swarm is shared among all swarms.
    # TODO: This can change in the future according to the implementation.

    def __init__(self, nr_particles : int, nr_dimensions : int,
                 objective_function, domain_low_boundaries : np.array, domain_high_boundaries : np.array,
                 maximum_iterations : int): 
        super.__init__()

        # ==================== INITIALIZE ATTRIBUTES FROM INPUT PARAMETERS START ====================
        self.__nr_particles = nr_particles
        self.__nr_dimensions = nr_dimensions
        # Initialize a random swarm of particles, represented by a 2D numpy array of dimensions [nr_particles, nr_dimensions]
        # Currently, the initialization is done using a uniform distribution.
        self.swarm_positions = np.random.default_rng().uniform(low=domain_low_boundaries, high=domain_high_boundaries)
        # Currently, the particles will start with zero velocity.
        self.swarm_velocities = np.zeros(shape=(nr_particles, nr_dimensions))

        self.__objective_function = objective_function

        self.__maximum_iterations = maximum_iterations
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
        objective_values = self.__objective_function(self.swarm)

        # STEP 2a
        best_objective_value_index = np.argmin(objective_values)

        # STEP 3a
        self.gbest_value = objective_values[best_objective_value_index]
        self.gbest_position = np.copy(self.swarm_positions[best_objective_value_index])
        #! Make sure to copy by value. Simple indexing copies by reference (view).
        #! For more information, read https://numpy.org/doc/stable/user/basics.copies.html


        # STEP 2b
        self.pbest_values = np.copy(objective_values)
        self.pbest_positions = np.copy(self.swarm)
        #! Make sure to copy by value. Simple indexing copies by reference (view).
        #! For more information, read https://numpy.org/doc/stable/user/basics.copies.html
        # ==================== INITIALIZE ATTRIBUTES FROM INTERNAL FUNCTIONS START ====================




        # ==================== INITIALIZE ATTRIBUTES ARBITRARILY START ====================
        self.w = 1.0
        self.c1, self.c2 = 2.0, 2.0
        # ==================== INITIALIZE ATTRIBUTES FROM ARBITRARILY FINISH ====================
    
    def step(self) -> None:
        self.step_velocities()
        # Translate the swarm positions according to the updated velocities.
        self.swarm_positions += self.swarm_velocities

    def step_velocities(self) -> None:
        random_generator = np.random.default_rng()
        R1 = np.diag(random_generator.uniform(size=self.__nr_particles * self.__nr_dimensions))
        R2 = np.diag(random_generator.uniform(size=self.__nr_particles * self.__nr_dimensions))
        # Reshaping the swarm matrix into a single (column) vector to utilize vectorization of numpy.
        # Both representations (3D matrix x 2D matrix, larger 2D matrix, large 1D vector) are equivalent.
        # NOTE: Because we are multiplying with random values, whether we do matrix-wise multiplication (@)
        # or element-wise multiplication (*) should be identical.
        # Element-wise multiplication is used, because it is believed to be computationally cheaper.
        cognitive_velocities = (R1 * (self.pbest_positions - self.swarm_positions).flatten()).reshape(self.__nr_particles, self.__nr_dimensions)
        social_velocities = (R2 * (self.gbest_position - self.swarm_positions).flatten()).reshape(self.__nr_particles, self.__nr_dimensions)

        self.velocities = self.w * self.swarm_velocities + self.c1 * cognitive_velocities + self.c2 * social_velocities

    def get_swarm_centroid(self):
        return np.mean(self.swarm, axis=0)