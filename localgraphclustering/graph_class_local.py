from .interface.types.graph import Graph
import networkx as nx
import csv
from scipy import sparse as sp
import scipy.sparse.linalg as splinalg
import numpy as np


class GraphLocal(Graph):
    """
    This class implements graph loading from an edgelist or gml and provides methods that operate on the graph.

    Attributes
    ----------
    adjacency_matrix : scipy csr matrix

    _num_vertices : int
        Number of vertices

    _num_edges : int
        Number of edges

    _directed : boolean
        Declares if it is a directed graph or not

    _weighted : boolean
        Declares if it is a weighted graph or not

    _dangling_nodes : int numpy array
        Nodes with zero edges

    d : float64 numpy vector
        Degrees vector

    dn : float64 numpy vector
        Component-wise reciprocal of degrees vector

    d_sqrt : float64 numpy vector
        Component-wise square root of degrees vector

    dn_sqrt : float64 numpy vector
        Component-wise reciprocal of sqaure root degrees vector

    vol_G : float64 numpy vector
        Volume of graph

    components : list of sets
        Each set contains the indices of a connected component of the graph

    number_of_components : int
        Number of connected components of the graph

    bicomponents : list of sets
        Each set contains the indices of a biconnected component of the graph

    number_of_bicomponents : int
        Number of connected components of the graph

    core_numbers : dictionary
        Core number for each vertex

    Methods
    -------
    import_text(filename, separator)
        Imports text from file.

    read_graph(fname, dtype='gml', separator='\t')
        Reads the graph from a file

    compute_statistics()
        Computes statistics for the graph

    connected_components()
        Computes the connected components of the graph

    is_disconnected()
        Checks if graph is connected

    biconnected_components():
        Computes the biconnected components of the graph

    core_number()
        Returns the core number for each vertex

    import_text(filename, separator)
        Reads text from filename
    """
    def __init__(self, filename, file_type='gml', separator='\t'):
        """
        Initializes the graph from a gml or a edgelist file and initializes the attributes of the class.

        Parameters
        ----------
        filename : string
            Name of the file, for example 'JohnsHopkins.edgelist' or 'JohnsHopkins.gml'.

        dtype : string
            Type of file. Currently only 'edgelist' and 'gml' are supported.
            Default = 'gml'

        separator : string
            used if file_type = 'edgelist'
            Default = '\t'
        """
        super().__init__(filename,file_type,separator)

        self.read_graph(filename, file_type, separator)

    def import_text(self, filename, separator):
        """
        Reads text from filename.
        """
        for line in csv.reader(open(filename), delimiter=separator, skipinitialspace=True):
            if line:
                yield line

    def read_graph(self, filename, file_type='gml', separator='\t'):
        """
        Reads the graph from a gml or a edgelist file and initializes the class attribute adjacency_matrix.

        Parameters
        ----------
        filename : string
            Name of the file, for example 'JohnsHopkins.edgelist' or 'JohnsHopkins.gml'.

        dtype : string
            Type of file. Currently only 'edgelist' and 'gml' are supported.
            Default = 'gml'

        separator : string
            used if file_type = 'edgelist'
            Default = '\t'
        """
        if file_type == 'edgelist':
        
            first_column = []
            second_column = []

            for data in self.import_text(filename, separator):
                first_column.extend([int(data[0])])
                second_column.extend([int(data[1])])

            if len(first_column) != len(second_column):
                print('The edgelist input is corrupted')

            self._num_edges = len(first_column)
            m = self._num_edges
            self._num_vertices = max([max(second_column),max(first_column)]) + 1
            n = self._num_vertices

            self.adjacency_matrix = sp.coo_matrix((np.ones(m),(first_column,second_column)), shape=(n,n))
            self.adjacency_matrix = self.adjacency_matrix.tocsr()
            self.adjacency_matrix = self.adjacency_matrix + self.adjacency_matrix.T
        elif file_type == 'gml':
            G = nx.read_gml(filename)
            self.adjacency_matrix = nx.adjacency_matrix(G).astype(np.float64)
            self._num_edges = nx.number_of_edges(G)
            self._num_vertices = nx.number_of_nodes(G)
        else:
            print('This file type is not supported')
            return
        
        self.compute_statistics()

    def compute_statistics(self):
        """
        Computes statistics for the graph. It updates the class attributes. The user needs to read the graph first before calling
        this method by calling the read_graph method from this class.
        """
        n = self._num_vertices
        
        self.d = np.ravel(self.adjacency_matrix.sum(axis=1))
        self._dangling = np.where(self.d == 0)[0]
        if self._dangling.shape[0] > 0:
            print('The following nodes have no outgoing edges:',self._dangling,'\n')
            print('These nodes are stored in the your_graph_object.dangling.')
            print('To avoid numerical difficulties we connect each dangling node to another randomly chosen node.')
            
            self.adjacency_matrix = sp.lil_matrix(self.adjacency_matrix)
            
            for i in self._dangling:
                numbers = list(range(0,i))+list(range(i + 1,n - 1))
                j = np.random.choice(numbers)
                self.adjacency_matrix[i,j] = 1
                self.adjacency_matrix[j,i] = 1
                
            self.adjacency_matrix = sp.csr_matrix(self.adjacency_matrix)

            self.d = np.ravel(self.adjacency_matrix.sum(axis=1))
        
        self.dn = 1.0/self.d
        self.d_sqrt = np.sqrt(self.d)
        self.dn_sqrt = np.sqrt(self.dn)
        self.vol_G = np.sum(self.d)

    def connected_components(self):
        """
        Computes the connected components of the graph. It stores the results in class attributes components
        and number_of_components. The user needs to call read the graph
        first before calling this function by calling the read_graph function from this class.
        This function calls Networkx.
        """
        g_nx = nx.from_scipy_sparse_matrix(self.adjacency_matrix)

        self.components = list(nx.connected_components(g_nx))

        self.number_of_components = nx.number_connected_components(g_nx)
        
        print('There are ', self.number_of_components, ' connected components in the graph') 

    def is_disconnected(self):
        """
        The output can be accessed from the graph object that calls this function.

        Checks if the graph is a disconnected graph. It prints the result as a comment and
        returns True if the graph is disconnected, or false otherwise. The user needs to
        call read the graph first before calling this function by calling the read_graph function from this class.
        This function calls Networkx.

        Returns
        -------
        True
             If connected

        False
             If disconnected
        """
        if self.d == []:
            print('The graph has to be read first.')
            return
        
        self.connected_components()
        
        if self.number_of_components > 1:
            print('The graph is a disconnected graph.')
            return True
        else: 
            print('The graph is not a disconnected graph.')
            return False

    def biconnected_components(self):
        """
        Computes the biconnected components of the graph. It stores the results in class attributes bicomponents
        and number_of_bicomponents. The user needs to call read the graph first before calling this
        function by calling the read_graph function from this class. This function calls Networkx.
        """
        g_nx = nx.from_scipy_sparse_matrix(self.adjacency_matrix)

        self.bicomponents = list(nx.biconnected_components(g_nx))
        
        self.number_of_bicomponents = len(self.bicomponents)

    def core_number(self):
        """
        Returns the core number for each vertex. A k-core is a maximal
        subgraph that contains nodes of degree k or more. The core number of a node
        is the largest value k of a k-core containing that node. The user needs to
        call read the graph first before calling this function by calling the read_graph
        function from this class. The output can be accessed from the graph object that
        calls this function. It stores the results in class attribute core_numbers.
        """
        g_nx = nx.from_scipy_sparse_matrix(self.adjacency_matrix)

        self.core_numbers = nx.core_number(g_nx)
