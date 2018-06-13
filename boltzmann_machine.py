import itertools

from activation_function import Activation_Function

from boltzmann_machine_node import Boltzmann_Machine_Node

import numpy as np

from undirected_graph import Undirected_Graph



class Boltzmann_Machine(Undirected_Graph):
    def __init__(self, X, order=1, hidden_nodes=0, sigma=Activation_Function('logistic'), random_initialisation=True):
        """
        Parameters:
        -----------
        X: numpy.ndarray
            A matrix in a numpy array, where the columns represent the visiable
            node and the rows represents the observations.
        order: int
            The order of the Boltzmann Machine
        hidden_nodes: int
            The number of higher order nodess.
            will be 20 higher order nodes in each layer.
        random_initialisation: bool
            If set to True, the bias and weights are randomly initialised
            If set to False, the weights and bias are initialised as 0

        Returns:
        --------
        None
        """
        super().__init__()

        self._order = order
        self._read_data(X)
        self._sigma = sigma
        self._Z = 0
        N = X.shape[0]

        # Create nodes new layers
        count = 0
        num_visible_nodes = X.shape[1]
        num_higher_order_nodes = hidden_nodes
        while(count < num_higher_order_nodes):
            h = (count // num_visible_nodes) + 1
            s = self._create_node()
            s.set_visibility(False)
            s.set_order(h)
            i = 0
            while(i < N):
                s.X.append(0)
                s.X_pred.append(0)
                i += 1
            count += 1

        # Connect all nodes
        self._fully_connect_nodes()
        self._update_vectors()
        self._calculate_empirical_distribution()
        self._update_nodes()

    def _create_node(self, x=0, name=None):
        """
        Creates a node on the Boltzmann Machine

        Parameters:
        -----------
        x: int
            Observed Value, could be 0 or 1.
        name: str
            Name of the node. Should not be a number
            This parameter is optional

        Returns:
        --------
        s: Boltzmann_Machine_Node
            Returns the newly created node.
        """
        name = self.create_vertex(name=None, Vertex=Boltzmann_Machine_Node)
        s = self.get_vertex(name)
        s.x = x
        s.b = np.random.normal(loc=0,scale=0.01)
        s.p = np.random.rand()

        return s

    def _read_data(self, X):
        """
        Reads in dataset

        Parameters:
        -----------
        X: numpy.ndarray
            A 2D matrix, where the rows are a new data point
            and the columns represent a new node.

        Returns:
        --------
        None
        """
        N = X.shape[0]
        N_nodes = X.shape[1]

        h = 0  # order of the node
        # Create nodes
        while(len(self.vertices) < N_nodes):
            s = self._create_node()
            s.set_visibility(True)
            s.set_order(h)
            i = 0
            while(i < N):
                s.X_pred.append(0)
                i += 1

        # Assign values
        for i, n in enumerate(self.vertices):
            n.X = X[:, i]

    def _fully_connect_nodes(self):
        S = self.vertices

        for i, si in enumerate(S):
            for j, sj in enumerate(S):
                if((si.h + 1) == sj.h):
                    self.add_edge(si.node_id, sj.node_id, np.random.normal(loc=0,scale=0.01))
                    si.p_edge_weights_emp.append(0)
                    sj.p_edge_weights_emp.append(0)

    def _update_hidden_states(self):
        """
        Calculate hidden state from visible state.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        hidden_idx = self.h_vec == 1
        hidden_bias = self.b_vec[hidden_idx]
        neigbhours_activation = np.array([np.sum(w[n == 1] * self.x_vec[n == 1])
                for n, w, h
                in zip(self.neighbours_vec, self.edge_weights_vec, self.h_vec)
                if h == 1])
        hidden_activation = hidden_bias + neigbhours_activation
        self.p_vec[hidden_idx] = self.sigma.apply(hidden_activation)
        self.x_vec[hidden_idx] = self.p_vec[hidden_idx] > np.random.rand(np.sum(hidden_idx))

    def _update_visible_states(self):
        """
        Reconstruct visible state from hidden state.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        visible_idx = self.h_vec == 0
        visible_bias = self.b_vec[visible_idx]
        neigbhours_activation = np.array([np.sum(w[n == 1] * self.x_vec[n == 1])
                for n, w, h
                in zip(self.neighbours_vec, self.edge_weights_vec, self.h_vec)
                if h == 0])
        hidden_activation = visible_bias + neigbhours_activation
        self.p_vec[visible_idx] = self.sigma.apply(hidden_activation)
        self.x_vec[visible_idx] = self.p_vec[visible_idx] > np.random.rand(np.sum(visible_idx))

    def _calculate_empirical_distribution(self):
        """
        Calculate the empirical distribution.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """

        for i, X in enumerate(self.X_vec.T):
            self.x_vec = X
            self._update_hidden_states()
            self.X_vec[:, i] = self.x_vec

        self.p_b_emp_vec = np.array([np.sum(X) / len(X) for X in self.X_vec])

        for i, n in enumerate(self.neighbours_vec):
            self.p_edge_weights_emp_vec[n == 1, i] = (
                [np.sum(self.X_vec[i, :] * X) / len(X)
                for X in self.X_vec[n == 1, :]])

    def _gradient_descent_step(self, lr=0.01, n_gibbs_iter=2, update_vectors=True):
        """
        A single iteration of gradient descent.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        if update_vectors:
            self._update_vectors()

        self._calculate_empirical_distribution()
        N = self.X_vec.shape[1]

        for i, X in enumerate(self.X_vec.T):
            self.x_vec = X

            # Gibbs sampling to get model prediction
            for _ in range(n_gibbs_iter):
                self._update_hidden_states()
                self._update_visible_states()

            self.X_pred_vec[:, i] = self.x_vec

        weight_gradient = self.neighbours_vec \
            * (np.dot(self.X_vec, self.X_vec.T) \
            - np.dot(self.X_pred_vec, self.X_pred_vec.T)) / N

        self.edge_weights_vec += lr * weight_gradient

        bias_gradient = self.p_b_emp_vec - self.p_vec
        self.b_vec += lr * bias_gradient

        error = (np.sum(weight_gradient**2) + np.sum(bias_gradient**2))**0.5

        if update_vectors:
            self._update_nodes()

        return error

    def train(self, tol=1e-7, lr=0.01, n_gibbs_iter=2, MAX_ITERATIONS=1000, verbose=False):
        """
        Run the model until the empirical distribution
        and the model distribution have reached the
        error threshold.

        Parameters:
        -----------
        tol: float
            The error threshold

        Returns:
        --------
        error: float
            The error value
        """
        self._update_vectors()

        iterations = 0
        conv_flag = True
        while(conv_flag):
            error = self._gradient_descent_step(lr=lr, n_gibbs_iter=n_gibbs_iter, update_vectors=False)
            iterations = iterations + 1

            if verbose:
                print('iterations:', iterations, 'error:', error)

            if error < tol:
                conv_flag = False

            if MAX_ITERATIONS < iterations:
                print('Did not converge. Maximum iterations reached.')
                break

        self._update_nodes()
        return error


    def _clique_potential_energy(self):
        """
        Calculate the energy of the current configuration
        of the graph.

        Parameters:
        -----------
        None

        Returns:
        --------
        phi: float
            The energy of the entire graph.
        """
        phi = 0
        # Calculate potential from bias
        for v in self.vertices:
            phi -= v.b * v.x

        # Calculate potential from edge weights
        for e in self.edges:
            w = e[0].get_edge_weight(e[1])
            phi -= w * e[0].x * e[1].x

        return phi

    def _calculate_normaliser(self):
        """
        Calculate the normalisation factor of the graph.
        This is ensure that the probability sums to one.
        This is done by calculating all possibility of the
        Boltzmann Machine.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        S = self.vertices
        N = len(S)

        # Create all possible combinations of the Boltzmann Machine
        possible_values = [0, 1]
        unique_combinations = itertools.product(possible_values, repeat=N)

        # Iterate through each unique combination of the Boltzmann Machine
        Z = 0
        for C in unique_combinations:
            phi = 0
            # Assign temporary value
            for s, c in zip(S, C):
                s.x = c

            phi = self._clique_potential_energy()

            # Add this combination of clique potential to Z
            Z += np.exp(-phi)

        return Z

    def boltzmann_machine_probability(self):
        """
        The probability of the Boltzmann Machine

        Parameters:
        -----------
        None

        Returns:
        --------
        p: float
            The probability of the Boltzmann machine.
        """
        S = self.vertices

        # Evaulate probability
        phi = self._clique_potential_energy()
        p = np.exp(-phi) / self.Z

        return p

    def predict(self, X):
        """

        """
        N = X.shape[0]
        S = self.vertices
        num_hidden_nodes = len(S) - X.shape[1]

        self.Z = self._calculate_normaliser()

        prediction = list()
        # Iterate through each test case
        for n in range(N):
            # Create all possible combinations of the Boltzmann Machine
            possible_values = [0, 1]
            unique_combinations = itertools.product(possible_values, repeat=num_hidden_nodes)

            p = 0
            for C in unique_combinations:
                c = iter(C)
                # Assign values to each node
                for i, s in enumerate(S):
                    if s.is_visible:
                        s.x = X[n, i]
                    else:
                        s.x = next(c)

                p += self.boltzmann_machine_probability()

            prediction.append(p)

        return prediction

    def _update_vectors(self):
        """

        """
        S = self.vertices

        self.x_vec = np.array([s.x for s in S])
        self.X_vec = np.array([s.X for s in S])
        self.X_pred_vec = np.array([s.X_pred for s in S])
        self.h_vec = np.array([s.h for s in S])
        self.is_visible_vec = np.array([s.is_visible for s in S])

        self.p_vec = np.array([s.p for s in S])

        # self.p_b_vec = np.array([s.p_b for s in S])
        self.p_b_emp_vec = np.array([s.p_b_emp for s in S])

        self.b_vec = np.array([s.b for s in S])

        N = len(S)
        self.neighbours_vec = np.zeros((N, N))
        self.edge_weights_vec = np.zeros((N, N))
        self.p_edge_weights_emp_vec = np.zeros((N, N))

        for s in S:
            for n, w, e_emp in zip(s.neighbours, s.edge_weights, s.p_edge_weights_emp):
                self.neighbours_vec[s.name, n.name] = 1
                self.edge_weights_vec[s.name, n.name] = w
                self.p_edge_weights_emp_vec[s.name, n.name] = e_emp

    def _update_nodes(self):
        """

        """
        S = self.vertices

        for i, s in enumerate(S):
            s.x = self.x_vec[i]
            s.X = self.X_vec[i]
            s.X_pred = self.X_pred_vec[i]
            # s.h = self.h_vec[i]
            s.is_visible = self.is_visible_vec[i]

            s.p = self.p_vec[i]

            # s.p_b = self.p_b_vec[i]
            s.p_b_emp = self.p_b_emp_vec[i]

            idx = self.neighbours_vec[i] == 1
            s.neighbours = [S[i] for i in np.where(idx)[0]]
            s.edge_weights = list(self.edge_weights_vec[i][idx])
            s.p_edge_weights_emp = list(self.p_edge_weights_emp_vec[i][idx])

    """
    Setter and getters
    """

    @property
    def sigma(self):
        return self._sigma

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, Z):
        self._Z = Z
