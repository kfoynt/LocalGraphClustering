import networkx as nx
import csv
from scipy import sparse as sp
from scipy.sparse import csgraph
import scipy.sparse.linalg as splinalg
import numpy as np
import pandas as pd
import warnings
import collections as cole
from .find_library import *

import gzip
import bz2
import lzma


class GraphLocal:
    """
    This class implements graph loading from an edgelist, gml or graphml and provides methods that operate on the graph.

    Attributes
    ----------
    adjacency_matrix : scipy csr matrix

    ai : numpy vector
        CSC format index pointer array, its data type is determined by "itype" during initialization

    aj : numpy vector
        CSC format index array, its data type is determined by "vtype" during initialization

    _num_vertices : int
        Number of vertices

    _num_edges : int
        Number of edges

    _weighted : boolean
        Declares if it is a weighted graph or not

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

    read_graph(filename, file_type='edgelist', separator='\t')
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

    neighbors(vertex)
        Returns a list with the neighbors of the given vertex

    list_to_gl(source,target)
        Create a GraphLocal object from edge list
    """
    def __init__(self,
        filename = None,
        file_type='edgelist',
        separator='\t',
        remove_whitespace=False,header=False, headerrow=None,
        vtype=np.uint32,itype=np.uint32):
        """
        Initializes the graph from a gml or a edgelist file and initializes the attributes of the class.

        Parameters
        ----------
        See read_graph for a description of the parameters.
        """

        if filename != None:
            self.read_graph(filename, file_type = file_type, separator = separator, remove_whitespace = remove_whitespace,
                header = header, headerrow = headerrow, vtype=vtype, itype=itype)

        self.load_library()

    def load_library(self):
        self.lib = load_library()
        return is_loaded(self.lib._name)

    def reload_library(self):
        self.lib = reload_library(self.lib)

    def read_graph(self, filename, file_type='edgelist', separator='\t', remove_whitespace=False, header=False, headerrow=None, vtype=np.uint32, itype=np.uint32):
        """
        Reads the graph from an edgelist, gml or graphml file and initializes the class attribute adjacency_matrix.

        Parameters
        ----------
        filename : string
            Name of the file, for example 'JohnsHopkins.edgelist', 'JohnsHopkins.gml', 'JohnsHopkins.graphml'.

        file_type : string
            Type of file. Currently only 'edgelist', 'gml' and 'graphml' are supported.
            Default = 'edgelist'

        separator : string
            used if file_type = 'edgelist'
            Default = '\t'

        remove_whitespace : bool
            set it to be True when there is more than one kinds of separators in the file
            Default = False

        header : bool
            This lets the first line of the file contain a set of heade
            information that should be ignore_index
            Default = False

        headerrow : int
            Use which row as column names. This argument takes precidence over
            the header=True using headerrow = 0
            Default = None

        vtype
            numpy integer type of CSC format index array
            Default = np.uint32

        itype
            numpy integer type of CSC format index pointer array
            Default = np.uint32
        """
        if file_type == 'edgelist':

            #dtype = {0:'int32', 1:'int32', 2:'float64'}
            if header and headerrow is None:
                headerrow = 0

            if remove_whitespace:
                df = pd.read_csv(filename, header=headerrow, delim_whitespace=remove_whitespace)
            else:
                df = pd.read_csv(filename, sep=separator, header=headerrow, delim_whitespace=remove_whitespace)
            cols = [0,1,2]
            if header != None:
                cols = list(df.columns)
            source = df[cols[0]].values
            target = df[cols[1]].values
            if df.shape[1] == 2:
                weights = np.ones(source.shape[0])
            elif df.shape[1] == 3:
                weights = df[cols[2]].values
            else:
                raise Exception('GraphLocal.read_graph: df.shape[1] not in (2, 3)')
            self._num_vertices = max(source.max() + 1, target.max()+1)
            #self.adjacency_matrix = source, target, weights

            self.adjacency_matrix = sp.csr_matrix((weights.astype(np.float64), (source, target)), shape=(self._num_vertices, self._num_vertices))

        elif file_type == 'gml':
            warnings.warn("Loading a gml is not efficient, we suggest using an edgelist format for this API.")
            G = nx.read_gml(filename).to_undirected()
            self.adjacency_matrix = nx.adjacency_matrix(G).astype(np.float64)
            self._num_vertices = nx.number_of_nodes(G)

        elif file_type == 'graphml':
            warnings.warn("Loading a graphml is not efficient, we suggest using an edgelist format for this API.")
            G = nx.read_graphml(filename).to_undirected()
            self.adjacency_matrix = nx.adjacency_matrix(G).astype(np.float64)
            self._num_vertices = nx.number_of_nodes(G)

        else:
            print('This file type is not supported')
            return
        self._weighted = False
        for i in self.adjacency_matrix.data:
            if i != 1:
                self._weighted = True
                break
        is_symmetric = (self.adjacency_matrix != self.adjacency_matrix.T).sum() == 0
        if not is_symmetric:
            # Symmetrize matrix, choosing larger weight
            sel = self.adjacency_matrix.T > self.adjacency_matrix
            self.adjacency_matrix = self.adjacency_matrix - self.adjacency_matrix.multiply(sel) + self.adjacency_matrix.T.multiply(sel)
            assert (self.adjacency_matrix != self.adjacency_matrix.T).sum() == 0

        self._num_edges = self.adjacency_matrix.nnz
        self.compute_statistics()
        self.ai = itype(self.adjacency_matrix.indptr)
        self.aj = vtype(self.adjacency_matrix.indices)

    def list_to_gl(self,source,target,weights,vtype=np.uint32, itype=np.uint32):
        """
        Create a GraphLocal object from edge list.

        Parameters
        ----------
        source
            A numpy array of sources for the edges

        target
            A numpy array of targets for the edges

        weights
            A numpy array of weights for the edges

        vtype
            numpy integer type of CSC format index array
            Default = np.uint32

        itype
            numpy integer type of CSC format index pointer array
            Default = np.uint32
        """
        self._num_edges = len(source)
        self._num_vertices = max(source.max() + 1, target.max()+1)
        self.adjacency_matrix = sp.csr_matrix((weights.astype(np.float64), (source, target)), shape=(self._num_vertices, self._num_vertices))
        self._weighted = False
        for i in self.adjacency_matrix.data:
            if i != 1:
                self._weighted = True
                break
        is_symmetric = (self.adjacency_matrix != self.adjacency_matrix.T).sum() == 0
        if not is_symmetric:
            # Symmetrize matrix, choosing larger weight
            sel = self.adjacency_matrix.T > self.adjacency_matrix
            self.adjacency_matrix = self.adjacency_matrix - self.adjacency_matrix.multiply(sel) + self.adjacency_matrix.T.multiply(sel)
            assert (self.adjacency_matrix != self.adjacency_matrix.T).sum() == 0

        self._num_edges = self.adjacency_matrix.nnz
        self.compute_statistics()
        self.ai = itype(self.adjacency_matrix.indptr)
        self.aj = vtype(self.adjacency_matrix.indices)

    def discard_weights(self):
        """ Discard any weights that were loaded from the data file.
        This sets all the weights associated with each edge to 1.0,
        which is our "no weight" case."""
        self.adjacency_matrix.data.fill(1.0)
        self._weighted = False
        self.compute_statistics()

    def compute_statistics(self):
        """
        Computes statistics for the graph. It updates the class attributes.
        The user needs to read the graph first before calling
        this method by calling the read_graph method from this class.
        """
        self.d = np.ravel(self.adjacency_matrix.sum(axis=1))
        self.dn = np.zeros(self._num_vertices)
        self.dn[self.d != 0] = 1.0 / self.d[self.d != 0]
        self.d_sqrt = np.sqrt(self.d)
        self.dn_sqrt = np.sqrt(self.dn)
        self.vol_G = np.sum(self.d)

    def connected_components(self):
        """
        Computes the connected components of the graph. It stores the results in class attributes components
        and number_of_components. The user needs to call read the graph
        first before calling this function by calling the read_graph function from this class.
        """

        output = csgraph.connected_components(self.adjacency_matrix,directed=False)

        self.components = output[1]
        self.number_of_components = output[0]

        #warnings.warn("Warning, connected_components is not efficiently implemented.")

        #g_nx = nx.from_scipy_sparse_matrix(self.adjacency_matrix)
        #self.components = list(nx.connected_components(g_nx))
        #self.number_of_components = nx.number_connected_components(g_nx)

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
        warnings.warn("Warning, biconnected_components is not efficiently implemented.")

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
        warnings.warn("Warning, core_number is not efficiently implemented.")

        g_nx = nx.from_scipy_sparse_matrix(self.adjacency_matrix)

        self.core_numbers = nx.core_number(g_nx)

    def neighbors(self,vertex):
        """
        Returns a list with the neighbors of the given vertex.
        """
        return self.adjacency_matrix[:,vertex].nonzero()[0].tolist()

    def compute_conductance(self,R):
        """
        Returns the conductance corresponding to set R.
        """
        v_ones_R = np.zeros(self._num_vertices)
        v_ones_R[R] = 1

        vol_R = sum(self.d[R])

        cut_R = vol_R - np.dot(v_ones_R,self.adjacency_matrix.dot(v_ones_R.T))
        vol = (1.0*min(vol_R,self.vol_G - vol_R))
        cond_R = cut_R/vol if vol != 0 else 0

        return cond_R

    def set_scores(self,R):
        """
        Return various metrics of a set of vertices.
        """

        voltrue = sum(self.d[R])
        v_ones_R = np.zeros(self._num_vertices)
        v_ones_R[R] = 1
        cut = voltrue - np.dot(v_ones_R,self.adjacency_matrix.dot(v_ones_R.T))

        voleff = min(voltrue,self.vol_G - voltrue)

        sizetrue = len(R)
        sizeeff = sizetrue
        if voleff < voltrue:
            sizeeff = self._num_vertices - sizetrue

        # remove the stuff we don't want returned...
        del R
        del self
        del v_ones_R

        edgestrue = voltrue - cut
        edgeseff = voleff - cut

        cond = cut / voleff if voleff != 0 else 1
        isop = cut / sizeeff

        # make a dictionary out of local variables
        return locals()

    def largest_component(self):
        self.connected_components()
        if self.number_of_components == 1:
            #self.compute_statistics()
            return self
        else:
            # find nodes of largest component
            counter=cole.Counter(self.components)
            maxccnodes = []
            what_key = counter.most_common(1)[0][0]
            for i in range(self._num_vertices):
                if what_key == self.components[i]:
                    maxccnodes.append(i)

            # biggest component by len of it's list of nodes
            #maxccnodes = max(self.components, key=len)
            #maxccnodes = list(maxccnodes)

            warnings.warn("The graph has multiple (%i) components, using the largest with %i / %i nodes"%(
                     self.number_of_components, len(maxccnodes), self._num_vertices))

            g_copy = GraphLocal()
            g_copy.adjacency_matrix = self.adjacency_matrix[maxccnodes,:].tocsc()[:,maxccnodes].tocsr()
            g_copy._num_vertices = len(maxccnodes) # AHH!
            g_copy.compute_statistics()
            g_copy._weighted = self._weighted
            dt = np.dtype(self.ai[0])
            itype = np.int64 if dt.name == 'int64' else np.uint32
            dt = np.dtype(self.aj[0])
            vtype = np.int64 if dt.name == 'int64' else np.uint32
            g_copy.ai = itype(g_copy.adjacency_matrix.indptr)
            g_copy.aj = vtype(g_copy.adjacency_matrix.indices)
            return g_copy


    def local_extrema(self,vals,strict=False,reverse=False):
        """
        Find extrema in a graph based on a set of values.

        Parameters
        ----------

        G: GraphLocal

        vals: Sequence[float]
            a feature value per node used to find the ex against each other, i.e. conductance

        strict: bool
            If True, find a set of vertices where vals(i) < vals(j) for all neighbors N(j)
            i.e. local minima in the space of the graph
            If False, find a set of vertices where vals(i) <= vals(j) for all neighbors N(j)
            i.e. local minima in the space of the graph

        reverse: bool
            if True, then find local maxima, if False then find local minima
            (by default, this is false, so we find local minima)

        Returns
        -------

        minverts: Sequence[int]
            the set of vertices

        minvals: Sequence[float]
            the set of min values
        """
        n = self.adjacency_matrix.shape[0]
        minverts = []
        ai = np.uint64(self.adjacency_matrix.indptr)
        aj = np.uint32(self.adjacency_matrix.indices)
        factor = 1.0
        if reverse:
            factor = -1.0
        for i in range(n):
            vali = factor*vals[i]
            lmin = True
            for nzi in range(ai[i],ai[i+1]):
                v = aj[nzi]
                if v == i:
                    continue # skip self-loops
                if strict:
                    if vali < factor*vals[v]:
                        continue
                    else:
                        lmin = False
                else:
                    if vali <= factor*vals[v]:
                        continue
                    else:
                        lmin = False

                if lmin == False:
                    break # break out of the loop

            if lmin:
                minverts.append(i)

        minvals = vals[minverts]

        return minverts, minvals
