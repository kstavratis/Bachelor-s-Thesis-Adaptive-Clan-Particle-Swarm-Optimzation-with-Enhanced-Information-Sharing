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

class PSOBackbone:
    """
    The "PSOBackone" class encompasses all functionalities essential for any Particle Swarm Optimization (PSO) algorithm.

    Attributes
    ----------
    `swarm_positions` : np.array
        shape == (nr_particles, nr_dimensions)\n
        A 2D array whose cells represent the positions of the swarm's particles.
        The rows index the particle ID, while the column the DOF id.

    `swarm_velocities` : np.array
        shape == (nr_particles, nr_dimensions)\n
        A 2D array whose cells represent the velocities of the swarm's particles.
        The rows index the particle ID, while the column the DOF id.

    `_objective_function` : function
        A (class <function>) pointer to the problem landscape, expressed in a mathematical formula, the PSO is tasked with solving.
        The function should be conditioned in such a way such that it produces (__nr_particles,) np.array outputs
        from a (nr_particles, nr_dimensions) 2D np.array. i.e.
        function(np.array(n,m)) -> np.array(n)

    `_domain_boundaries` : np.array
        shape == (nr_dimensions, 2) or (2,)\n
        A 2D array whose i-th row contain the lower (position 0) and higher (position 1) bounds of the search domain for the i-th dimension.
        In the case of a 1D np.array, the argument is expanded so as to accommodate for all domain dimensions. 

    `w` : float
        The w (ω) value is known as the "inertia weight". w # Math: \omega \in [0, 1]
        The w weight dictates how much of the previously attained velocity will be used in the next iteration of the algorithm
        (0 := not at all, 1 := fully).\n
        Default value is 1.0.

    `c1` : float
        The c1 value is known as "cognitive weight" or "greedy learning factor"
        Its value (in conjunction with c2) determines the behaviour of each particle of the swarm.
        As a rule of thumb, the higher the ratio c1/c2 is, the more local exploitation is encouraged.\n
        Default value is 2.0

    `c2` : float
        The c2 value is known as "social weight" or "social learning factor"
        Its value (in conjunction with c1) determines the behaviour of each particle of the swarm.
        As a rule of the thumb, the higher the ration c2/c1 is, the more global exploration is encouraged.\n
        Default value is 2.0
    
        

    Although the class `PSOBackbone` can be instantiated on its own, it is recommended that additional schemes are built on top of this class,
    such that desirable properties (exploration, convergence, stability) are achieved.
    The recommended way of building additional PSO schemes is by implementing them as Python "Mixins".
    """

    def __init__(self, nr_particles : int, nr_dimensions : int,
                 objective_function, domain_boundaries : np.array,
                 input_swarm_positions : np.array = None,
                 input_pbest_positions : np.array = None,
                 **kwargs : dict):
        """
        Parameters
        ----------
        nr_particles : int
            The number of particles that form the swarm.
            Equivalent to the rows of the `swarm_positions` structure.
            This parameter is ignored in the case where `input_swarm_positions` is provided.

        nr_dimensions : int
            The number of dimensions of the problem to be solved.
            Equivalent to the columns of the `swarm_positions` structure.
            This parameter is ignored in the case where `input_swarm_positions` is provided.

        objective_function : function
            A pointer to a function which describes the landscape of the problem to solve
            This function must be formatted in a way such that it accepts 2D arrays as input
            and extracts 1D arrays as the output.

        domain_boundaries : np.array
             A 1D or 2D array whose i-th row contain the lower (position 0) and higher (position 1) bounds of the search domain for the i-th dimension.
            In the case of a 1D np.array, the argument is expanded so as to accommodate for all domain dimensions.

        input_swarm_positions : np.array
            A (p, d) 2D array representing the swarm.
            - p are the number of particles of the swarm
            - d are the dimensions of the problem.

            The inclusion of this parameter overrules parameters `nr_particles` and `nr_dimensions`.\n
            Default value is `None`.
        
        input_pbest_positions : np.array
            A (p, d) 2D array representing custom pbest positions with which the particles will be initialized.
            - p are the number of particles of the swarm
            - d are the dimensions of the problem.

            NOTE: The inserted pbest positions are overridden in case that
            the particles' initialization positions yield a better (lower) objective value.


        """ 
        super().__init__(**kwargs)

        # ==================== INITIALIZE ATTRIBUTES FROM INPUT PARAMETERS START ====================
        self._objective_function = objective_function

        # Handle input swarm, in case it is provided.
        self.swarm_positions = None # Declaring the "self.swarm_positions" attribute.
        if input_swarm_positions is not None and input_swarm_positions.ndim == 2:
            self.swarm_positions = input_swarm_positions
            nr_particles, nr_dimensions = input_swarm_positions.shape # Changing the argument values, because they're used later.
        else:
            # Dynamic sanity checks.
            assert isinstance(nr_particles, int), f'Argument "nr_particles" is expected to be an integer. Received a {type(nr_particles)} instead.'
            assert isinstance(nr_dimensions, int), f'Argument "nr_dimensions" is expected to be an integer. Received a {type(nr_dimensions)} instead.'
            assert nr_particles > 0, f'Argument "nr_particles" must be natural (a.k.a positive integer) number. {nr_particles} was provided instead.'
            assert nr_dimensions > 0, f'Argument "nr_particles" must be natural (a.k.a positive integer) number. {nr_dimensions} was provided instead.'

            # Initialize a random swarm of particles, represented by a 2D numpy array of dimensions [nr_particles, nr_dimensions]
            # Currently, the initialization is done using a uniform distribution.
            self.swarm_positions = np.random.default_rng().uniform(low=domain_boundaries[:, 0], high=domain_boundaries[:, 1], size=(nr_particles, nr_dimensions))

        # Handle input pbest positions, in case they are provided.
        self.pbest_positions, self.pbest_values = None, None # Declaring the "pbest" attributes.
        self.gbest_position, self.gbest_value = None, None # Declaring the "gbest" attributes.
        if input_pbest_positions is not None and input_swarm_positions.ndim == 2:

            objective_values = self._objective_function(input_pbest_positions)
            best_objective_value_index = np.argmin(objective_values)

            self.pbest_positions = input_pbest_positions
            self.pbest_values = objective_values

            self.gbest_value = objective_values[best_objective_value_index]
            self.gbest_position = np.copy(self.pbest_positions[best_objective_value_index])
            

            nr_particles, nr_dimensions = input_pbest_positions.shape # Changing the argument values, because they're used later.


        # Currently, the particles will start with zero velocity.
        self.swarm_velocities = np.zeros(shape=(nr_particles, nr_dimensions))


        if domain_boundaries.ndim == 1:
            domain_boundaries = np.tile(domain_boundaries, (nr_dimensions, 1)) # Repeat the same limit for all domain dimensions.
        self._domain_boundaries = domain_boundaries

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
        objective_values = self._objective_function(self.swarm_positions)

        # STEP 2a
        best_objective_value_index = np.argmin(objective_values)

        # STEP 3a
        if self.gbest_position is not None: # Insert "if" branch if a `input_pbest_positions` was provided.
            self.__update_gbest(self.swarm_positions, objective_values)
        else:
            self.gbest_value = objective_values[best_objective_value_index]
            self.gbest_position = np.copy(self.swarm_positions[best_objective_value_index])
            #! Make sure to copy by value. Simple indexing copies by reference (view).
            #! For more information, read https://numpy.org/doc/stable/user/basics.copies.html


        # STEP 2b
        if self.pbest_positions is not None: # Insert "if" branch if a `input_pbest_positions` was provided.
            self.__update_pbest(self.swarm_positions, objective_values)
        else:
            self.pbest_values = np.copy(objective_values)
            self.pbest_positions = np.copy(self.swarm_positions)
            #! Make sure to copy by value. Simple indexing copies by reference (view).
            #! For more information, read https://numpy.org/doc/stable/user/basics.copies.html



        # ==================== INITIALIZE ATTRIBUTES FROM INTERNAL FUNCTIONS START ====================

        # ==================== Filtering (slowing down) the resulting velocities START ====================
        # Numerous methods to limit the velocities are proposed in the literature.
        # The current implementation assumes a simple, yet widespread technique: limiting the velocity to 20% of the search domain's range.
        # NOTE: Desired limits: ( −0.2 * max{ |𝑋𝑚𝑖𝑛𝑑|, |𝑋𝑚𝑎𝑥𝑑|} , 0.2 * max{|𝑋𝑚𝑖𝑛𝑑|,|𝑋𝑚𝑎𝑥𝑑|} ), ∀d∈D (D := problem domain)
        self.__maximum_speeds = 0.2 * np.abs(self._domain_boundaries).max(axis=1) # Extract the maximum speed per dimension
        self.__maximum_speeds = np.tile(self.__maximum_speeds, (2, 1)).T # Create a copy column of the maximum speeds.
        self.__maximum_speeds[:, 0] = -self.__maximum_speeds[:, 0] #  Allow both positive and negative speeds in the same dimension
        self.__maximum_speeds = np.tile(self.__maximum_speeds, (nr_particles, nr_dimensions // domain_boundaries.shape[0], 1)) # Expand that it limits all dimensions (done in previous commands) of ALL particles
        # NOTE: The final result is `self.__maximum_speeds[particle ID][dimension ID][2] = np.array(minimum of dimension, maximum of dimension)`.
        # TODO: Remove this as a private variable and develop different limiting methodologies. PRIORITY: LOW
        # ==================== Filtering (slowing down) the resulting velocities FINISH ====================



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
        self._update_weights_and_acceleration_coefficients()
        self._step_velocities()
        # Filtering (slowing down) the resulting velocities
        np.clip(self.swarm_velocities, self.__maximum_speeds[:, :, 0], self.__maximum_speeds[:, :, 1], out=self.swarm_velocities)
        # Translate the swarm positions according to the updated velocities.
        self.swarm_positions += self.swarm_velocities
        # Filtering the result. The current result manually forces the particles to stay in the domain.
        # However, in literature, numerous "wall" types have been proposed. Examples of names can be seen in "wall_types.py".
        np.clip(self.swarm_positions, self._domain_boundaries[:, 0], self._domain_boundaries[:, 1], out=self.swarm_positions)
        # TODO: Implement different kinds of walls.

        # Update the pbest and gbest of the swarm for the next step.
        self._update_pbest_and_gbest()

    def _step_velocities(self) -> None:
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
        R1 = random_generator.uniform(size=self.swarm_positions.shape)
        R2 = random_generator.uniform(size=self.swarm_positions.shape)
        cognitive_velocities = self.c1 * R1 * (self.pbest_positions - self.swarm_positions)
        social_velocities = self.c2 * R2 * (self.gbest_position - self.swarm_positions) # "gbest_position" is broadcast to shape of "swarm_positions"
        # ==================== Element-wise multiplication approach FINISH ====================

        self.swarm_velocities = self.w * self.swarm_velocities + cognitive_velocities + social_velocities

    # I have the function updating both, so as to evaluate the objective function only once, thus reducing computational time.
    def _update_pbest_and_gbest(self):
        objective_values = self._objective_function(self.swarm_positions)
        self.__update_pbest(self.swarm_positions, objective_values)
        self.__update_gbest(self.swarm_positions, objective_values)


    # def update_pbest(self):
    #     self.__update_pbest(self.swarm_positions, self._objective_function(self.swarm_positions))

    # def update_gbest(self):
    #     self.__update_gbest(self.swarm_positions, self._objective_function(self.swarm_positions))

    def __update_pbest(self, candidate_positions : np.array, candidate_objective_values : np.array) -> None:
        # For each particle,
        # in case that f(x) is lower than the currently memorized objective values "pbest_value"
        # update the personal best value and position

        # A boolean mask whose "True" values indicate which values will be replaced. NumPy allows boolean indexing.
        update_mask = candidate_objective_values < self.pbest_values

        self.pbest_values[update_mask] = candidate_objective_values[update_mask]
        self.pbest_positions[update_mask] = candidate_positions[update_mask] #* IMPORTANT: boolean indexing returns copy, not view.


    def __update_gbest(self, candidate_positions : np.array, candidate_objective_values : np.array) -> None:
        # STEP 1: Find the particle with the lowest value.
        best_objective_value_index = np.argmin(candidate_objective_values)

        # STEP 2: Update the global best value and position
        #           in case that f(x) is lower than the one currently memorized by the swarm "gbest_value".
        if candidate_objective_values[best_objective_value_index] < self.gbest_value:
            self.gbest_value = candidate_objective_values[best_objective_value_index]
            self.gbest_position = candidate_positions[best_objective_value_index].copy() #* IMPORTANT: Avoid numpy view.
        #! During testing, beware to check for copy by value or by reference! This can be done by checking the "base" attribute of np arrays.
        #! Make sure to copy by value. Simple indexing copies by reference (view).
        #! For more information, read https://numpy.org/doc/stable/user/basics.copies.html

    
    def _update_weights_and_acceleration_coefficients(self):
        # The definition of this function acts as a placeholder for any other PSO mixin which behaves in a particular manner.
        # For example, in the standard PSO, the "update_weights" function could refer to the linear decrease of the inertia weight ω.
        pass

    def get_reached_gbest_particle(self, random : bool = True):
        """
        Returns the particle which at some (previous) iteration reached the gbest of the swarm.

        Parameters
        ----------
        random : bool
            In cases where multiple particles reached the gbest position at some point,
            this value determines how the particle is chosen.
                True ->  One of the particles that had reached gbest is picked at random.
                False -> The first (index-wise) particle which satisfies the goal is returned.

        Returns
        -------
        : int
            Index (0-indexed) of the particle which at some point held the best (lowest) objective value

        : np.array
            A copy of the vector representing the particle which at some point held the best (lowest) objective value
            NOTE: shape = (nr_dimensions,)

        """

        index_of_gbest = np.where(np.all(self.pbest_positions == self.gbest_position, axis=1))[0] # Search particle which found gbest at some point.
        # There is the possibility that more than one particles have the same `pbest_positions`.
        # Such an example could be when more than one particles have reached the (local) optimum.
        if index_of_gbest.size > 1:
            if random:
                index_of_gbest = np.random.default_rng().choice(index_of_gbest) # Choose a random particle which reached gbest at some point.
            else:
                index_of_gbest = np.array([index_of_gbest[0]]) # Choose the first (index-wise) appearance/

        index_of_gbest = index_of_gbest.item()
        best_particle = self.swarm_positions[index_of_gbest].copy()

        return index_of_gbest, best_particle


    def get_current_best_particle(self):
        """
        Returns the particle which holds the best (lowest) objective value **at the current configuation**.
        This is not to be confused with `gbest_position`, which stores the position which held the best objective value
        at the current **or some previous** configuration!

        Returns
        -------
        : int
            Index (0-indexed) of the particle which holds the best (lowest) objective value
            *in the current configuration* of the swarm

        : np.array
            A copy of the vector representing the particle which holds the best (lowest) objective value
            *in the current configuration* of the swarm.
            NOTE: shape = (nr_dimensions,)

        : float
            Objective value of the particle which holds the best (lowest) objective value
            *in the current configuration* of the swarm.
        

        """
        objective_values = self._objective_function(self.swarm_positions)
        index_of_best_particle = np.argmin(objective_values)
        best_particle = self.swarm_positions[index_of_best_particle].copy()
        best_particle_value = objective_values[index_of_best_particle]

        return index_of_best_particle, best_particle, best_particle_value



    def get_swarm_centroid(self):
        return np.mean(self.swarm_positions, axis=0)
    
    def std_of_swarm_from_centroid(self):
        swarm_centroid = self.get_swarm_centroid()
        result = np.std(np.linalg.norm(swarm_centroid - self.swarm_positions, axis=1, ord=2))
        # TODO: See whether the return value is a scalar or an array.
        return result