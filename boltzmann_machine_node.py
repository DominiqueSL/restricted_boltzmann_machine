from vertex import Vertex

class Boltzmann_Machine_Node(Vertex):
    """
    Node containing the parameters for the Boltzmann machine
    """
    def __init__(self, node_id=None, visible_flag=True):
        """
        Parameters:
        -----------
        node_id: int
            Unique ID for the node

        Returns:
        --------
        None
        """
        super().__init__(node_id)

        self._node_id = node_id
        self._x = 0
        self._X = list()
        self._X_pred = list()
        self._h = 0
        self._is_visible = visible_flag

        self._p = 0

        # self._p_b = 0
        self._p_b_emp = 0

        # self._p_edge_weights = list()
        self._p_edge_weights_emp = list()

    def set_visibility(self, flag):
        """
        Sets whether the node is visible or hidden.
        By default the node is visible.

        Parameters:
        -----------
        flag: bool
            A boolean flag on whether the node is visiable
            If set to True, the node is visiable,
            If set to False, the node is hiddens.

        Returns:
        --------
        None
        """
        self.is_visible = flag

    def set_order(self, h):
        """
        Set which layer the node is in. h=1 represents the visiable
        and the hidden layer. h=2 represents the first higher order layer.

        Paramters:
        ----------
        h: int
            The order which the node belongs to

        Returns:
        --------
        None
        """
        self._h = h

    """
    Setters and getters
    """
    @property
    def node_id(self):
        return self._node_id

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X

    @property
    def X_pred(self):
        return self._X_pred

    @X_pred.setter
    def X_pred(self, X_pred):
        self._X_pred = X_pred

    @property
    def h(self):
        return self._h

    @property
    def is_visible(self):
        return self._is_visible

    @is_visible.setter
    def is_visible(self, is_visible):
        self._is_visible = is_visible

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p

    # @property
    # def p_b(self):
    #     return self._p_b

    # @p_b.setter
    # def p_b(self, p_b):
    #     self._p_b = p_b

    @property
    def p_b_emp(self):
        return self._p_b_emp

    @p_b_emp.setter
    def p_b_emp(self, p_b_emp):
        self._p_b_emp = p_b_emp

    # @property
    # def p_edge_weights(self):
    #     return self._p_edge_weights

    # @p_edge_weights.setter
    # def p_edge_weights(self, p_edge_weights):
    #     self._p_edge_weights = p_edge_weights

    @property
    def p_edge_weights_emp(self):
        return self._p_edge_weights_emp

    @p_edge_weights_emp.setter
    def p_edge_weights_emp(self, p_edge_weights_emp):
        self._p_edge_weights_emp = p_edge_weights_emp
