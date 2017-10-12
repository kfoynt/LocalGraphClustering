import abc


class Graph(metaclass=abc.ABCMeta):
    """
    This class implements graph loading from an edgelist or gml and
    provides an interface of methods that operate on the graph.

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

        self.adjacency_matrix = None
        self._num_vertices = None
        self._num_edges = None
        self._directed = None
        self._weighted = None
        self._dangling_nodes = None
        self.d = None
        self.dn = None
        self.d_sqrt = None
        self.dn_sqrt = None
        self.vol_G = None
        self.components = None
        self.number_of_components = None
        self.number_of_bicomponents = None
        self.bicomponents = None
        self.core_numbers = None

    @abc.abstractmethod
    def import_text(self, filename, separator):
        """
        Reads text from filename.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def compute_statistics(self):
        """
        Computes statistics for the graph. It updates the class attributes. The user needs to
        read the graph first before calling this method by calling the read_graph method from this class.
        """

    @abc.abstractmethod
    def connected_components(self):
        """
        Computes the connected components of the graph. It stores the results in class attributes
        components and number_of_components. The user needs to call read the graph first before
        calling this function by calling the read_graph function from this class.
        This function calls Networkx.
        """

    @abc.abstractmethod
    def is_disconnected(self):
        """
        The output can be accessed from the graph object that calls this function.

        Checks if the graph is a disconnected graph. It prints the result as a comment and
        returns True if the graph is disconnected, or false otherwise. The user needs to
        call read the graph first before calling this function by calling the read_graph
        function from this class. This function calls Networkx.

        Returns
        -------
        True
             If connected

        False
             If disconnected
        """

    @abc.abstractmethod
    def biconnected_components(self):
        """
        Computes the biconnected components of the graph. It stores the results in class attributes bicomponents
        and number_of_bicomponents. The user needs to call read the graph first before calling this
        function by calling the read_graph function from this class. This function calls Networkx.
        """

    @abc.abstractmethod
    def core_number(self):
        """
        Returns the core number for each vertex. A k-core is a maximal
        subgraph that contains nodes of degree k or more. The core number of a node
        is the largest value k of a k-core containing that node. The user needs to
        call read the graph first before calling this function by calling the read_graph
        function from this class. The output can be accessed from the graph object that
        calls this function. It stores the results in class attribute core_numbers.
        """
