import numpy as np

class Vertex(object):
    """
    This class provides functions for a vertex in an undirected graph
    """
    def __init__(self, vertex):
        """
        Parameters:
        -----------
        vertex: str
            Name of the vertex
        
        Returns:
        --------
        None
        """
        self._name = vertex
        self._neighbours = list()
        self._edge_weights = list()
        self._b = 0
        
    def get_edge_weight(self, neighbour):
        """
        Gets edge weight of neighbour
        
        Parameters:
        -----------
        neighbour: Vertex
            The vertex of the neighbour node
            
        Returns:
        --------
        w: float
            Edge weight between the two nodes.
        """
        idx = [i for i, n in enumerate(self.neighbours) if (n.name==neighbour.name)]
        w = np.array(self.edge_weights)[idx]
        return w
        
    def add_neighbour(self, neighbour, w=0, sort_neighbours=False):
        """
        Add neighbours to current vertex.
        
        Paramters:
        ----------
        neighbour: Vertex
            The ID of the vertex
            
        w: float
            Edge weight for two nodes in an undirected graph.
        Sort_neigbhours: bool
            If set to True, the neighbours are sorted when added.
            Default is True.
        
        Returns:
        --------
        None
            
        """        
        self.neighbours.append(neighbour)
        self.edge_weights.append(w)
        
        neighbour.neighbours.append(self)
        neighbour.edge_weights.append(w)
            


        # Sort neighbours if flag is set to True
        if sort_neighbours:
            self.sort_neigbhour_list(neighbour)
        
    def _sort_neighbour_list(self):
        """
        Sorts the neighbour list of current vertex.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        idx: numpy.ndarray
            The order of the sorted arguments.
        """
        idx = np.argsort([n.name for n in self.neighbours])
        self.neighbours = [self.neighbours[i] for i in idx]
        self.edge_weights = [self.edge_weights[i] for i in idx]

        return idx
            
    def sort_neigbhour_list(self, neighbour):
        """
        Sort the order of the neighbour list. This also sorts the neighbour
        list of the current vertex's neighbour.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
        """
        
        # Sort current vertex
        self._sort_neighbour_list()
        
        # Sort neighbouring vertex
        neighbour._sort_neighbour_list()
        
    def is_neighbour(self, vertex):
        """
        Checks if vertex is neighbour of this current vertex.
        
        Parameters:
        -----------
            Another vertex in this graph.
            
        Returns:
        --------
        flag: bool
            True if neighbour. False if not a neighbour.
        """
        neighbour_name = vertex.name
        flag = np.sum([n.name == neighbour_name for n in self.neighbours])
        return flag

    def set_bias(self, b):
        self.b = b

    """
    Setters and Getters
    """

    @property
    def name(self):
        """
        Gets the name of the vertex.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        name: int
            Returns the vertex ID.
        """
        return self._name
    
    @property
    def neighbours(self):
        """
        Returns a list of its neighbours
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        self.neigbhours: list
            Returns a list of its neighbours.
        """
        return self._neighbours
    
    @neighbours.setter
    def neighbours(self, neighbours):
        self._neighbours = neighbours
    
    @property
    def edge_weights(self):
        return self._edge_weights
    
    @edge_weights.setter
    def edge_weights(self, edge_weights):
            self._edge_weights = edge_weights

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        self._b = b