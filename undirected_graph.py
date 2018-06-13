import numpy as np
from vertex import Vertex

class Undirected_Graph(object):
    """
    This is an implementation of an undirected graph
    """
    def __init__(self):
        self._vertices = list()
        self._edges = list()
        self._vertex_name = dict()
        self._next_vertex_id = 0
        
    def get_vertex(self, name):
        return self.vertices[self.vertex_name[name]]
        
    def create_vertex(self, name=None, Vertex=Vertex):
        """
        Creates a vertex in the graph
        
        Paramters:
        ----------
        name: str
            The name of the node. Should not be
            a number to avoid confusion with node ID.
            This parameter is optional.
        Vertex: Vertex
            An object with class Vertex
        
        Returns:
        node_id: int
            A unique ID identifying the node
        """
        node_id = self.next_vertex_id
        if name == None:
            name = node_id
        
        self.vertices.append(Vertex(node_id))
        self.vertex_name[name] = node_id
        self.next_vertex_id += 1
        return node_id
        
    def add_edge(self, vertex_source, vertex_dest, w=0):
        """
        Adds an edge between two nodes. This is an
        undirected graph. So this will add an edge
        to both the source and destination.
        
        Parameters:
        -----------
        vertex_source: str
            The name of your node. If you did not
            provide a name for your node, then this
            is your node ID.
        vertex_dest: str
            The name of your node. If you did not
            provide a name for your node, then this
            is your node ID.
        w: float
            Edge weight for an undirected graph.
        Returns:
        --------
        None
    
        """
        v_s = self.vertices[self.vertex_name[vertex_source]]
        v_d = self.vertices[self.vertex_name[vertex_dest]]
        v_s.add_neighbour(v_d, w)
        self.edges.append((v_s, v_d, w))
        
    def print_graph(self):
        """
        This function returns a list of each of the nodes
        along with all its edges.

        Parameters:
        -----------
        None

        Returns:
        --------
        graph_list: list
            A list of all the nodes in the graph with the
            corrosponding edges.
        """
        graph_list = list()
        for v in self.vertices:
            graph_list.append([v.name, [(n.name, w) for n, w in zip(v.neighbours, v.edge_weights)]])
        
        return graph_list
            
    """
    Setters and getters
    """
    @property    
    def vertices(self):
        return self._vertices
    
    @property
    def edges(self):
        return self._edges
    
    @property
    def vertex_name(self):
        return self._vertex_name
    
    @property
    def next_vertex_id(self):
        return self._next_vertex_id
    
    @next_vertex_id.setter
    def next_vertex_id(self, next_vertex_id):
        self._next_vertex_id = next_vertex_id
        
    if __name__ == '__main__':
        g = Undirected_Graph()
        vertex_list = [g.create_vertex() for _ in range(4)]
        g.add_edge(0,2)
        g.add_edge(0,3)
        g.add_edge(1,2)
        g.add_edge(1,3)
        g.add_edge(2,3)
        
        graph_list = g.print_graph()
        for n in graph_list:
            print(n)
        print([(e[0].name, e[1].name) for e in g.edges])