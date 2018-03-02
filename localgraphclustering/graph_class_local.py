from .interface.types.graph import Graph
import networkx as nx
import csv
from scipy import sparse as sp
from scipy.sparse import csgraph
import scipy.sparse.linalg as splinalg
import numpy as np
import warnings
import collections as cole
from localgraphclustering.find_library import *

import gzip
import bz2
import lzma


class GraphLocal(Graph):
    """
    This class implements graph loading from an edgelist, gml or graphml and provides methods that operate on the graph.

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

    import_text(filename, separator)
        Reads text from filename
    """
    def __init__(self, filename = None, file_type='edgelist', separator='\t'):
        """
        Initializes the graph from a gml or a edgelist file and initializes the attributes of the class.

        Parameters
        ----------
        filename : string
            Name of the file, for example 'JohnsHopkins.edgelist' or 'JohnsHopkins.gml'.
            Default = 'None'

        dtype : string
            Type of file. Currently only 'edgelist' and 'gml' are supported.
            Default = 'edgelist'

        separator : string
            used if file_type = 'edgelist'
            Default = '\t'
        """
        super().__init__(filename,file_type,separator)

        if filename != None:
            self.read_graph(filename, file_type, separator)
        self.load_library()

    def load_library(self):
        self.lib = load_library()
        return is_loaded(self.lib._name)

    def reload_library(self):
        self.lib = reload_library(self.lib)

    def import_text(self, filename, separator):
        """
        Reads text from filename.
        """
        if filename.endswith(".gz"):
            fh = gzip.open(filename, "rt")
        elif filename.endswith(".bz2"):
            fh = bz2.open(filename, "rt")
        elif filename.endswith(".xz"):
            fh = lzma.open(filename, "rt")
        else:
            fh = open(filename, "rt")
            
        for line in csv.reader(fh, delimiter=separator, skipinitialspace=True):
            if line:
                yield line
                
        fh.close()
                
    def list_to_CSR(self,ei,ej,ev):
        """
        This function takes an edge list and returns it in csr_matrix format.

        Parameters
        ----------
        ei : numpy array
            a numpy array of the source nodes of edges
        ej : numpy array
            a numpy array of the dest nodes of edges
        ev : numpy array
            a numpy array of the weight of edges

        """
        n = max(max(ei),max(ej))+1
        m = len(ei)
        ai = np.zeros(n+1,dtype=np.int32)
        for i in ei:
            ai[i+1] += 1
        ind = 0
        for i in range(n):
            ai[i+1] += ai[i] 
        while ind < m:
            if ev[ind] < 0:
                ind += 1
            else:
                index = ei[ind]
                dest = ai[index]
                if dest == ind:
                    ev[dest] *= -1
                    ai[index] += 1
                    continue
                temp_ei,temp_ej,temp_ev = ei[dest],ej[dest],abs(ev[dest])
                ei[dest],ej[dest],ev[dest] = ei[ind],ej[ind],-ev[ind]
                ei[ind],ej[ind],ev[ind] = temp_ei,temp_ej,temp_ev
                ai[index] += 1
        for i in range(m):
            ev[i] = abs(ev[i])
        dummy = 0
        for i in range(n+1):
            temp = ai[i]
            ai[i] -= (ai[i]-dummy)
            dummy = temp
        M = sp.csr_matrix((ev, ej, ai), shape=(n, n))
        M.sort_indices()
        return M

    def check_symmetry(self,filename,separator):
        counter = 0
        another_count = 0
        test_set = set()
        not_checked = 0
        added = 0
        for data in self.import_text(filename, separator):
            counter += 1
            another_count += 1
            ei = int(data[0])
            ej = int(data[1])
            if ei != ej:
                another_count += 1
                added += 1
                if added <= 200:
                    test_set.add((ei,ej))
                    if (ej,ei) not in test_set:
                        not_checked += 1
                    else:
                        not_checked -= 1
                else:
                    if (ej,ei) in test_set:
                        not_checked -= 1
        if not_checked > 0:
            return (another_count,False)
        else:
            return (counter,True)

    def read_graph(self, filename, file_type='edgelist', separator='\t', symmetry_check=True):
        """
        Reads the graph from an edgelist, gml or graphml file and initializes the class attribute adjacency_matrix.

        Parameters
        ----------
        filename : string
            Name of the file, for example 'JohnsHopkins.edgelist', 'JohnsHopkins.gml', 'JohnsHopkins.graphml'.

        dtype : string
            Type of file. Currently only 'edgelist', 'gml' and 'graphml' are supported.
            Default = 'edgelist'

        separator : string
            used if file_type = 'edgelist'
            Default = '\t'
        """ 
        if file_type == 'edgelist':

            counter,is_complete = self.check_symmetry(filename,separator)
                    
            source = np.zeros(counter,int)
            target = np.zeros(counter,int)
            weights = np.zeros(counter,float)
            
            counter = 0
            
            if is_complete:
                for data in self.import_text(filename, separator):
                
                    if len(data) <= 2:
                        source[counter] = int(data[0])
                        target[counter] = int(data[1])
                        weights[counter] = 1
                    else:
                        source[counter] = int(data[0])
                        target[counter] = int(data[1])
                        weights[counter] = float(data[2])
                    
                    counter += 1
            else:
                for data in self.import_text(filename, separator):
                    ei,ej = int(data[0]),int(data[1])
                    if len(data) <= 2:
                        source[counter] = ei
                        target[counter] = ej
                        weights[counter] = 1
                        counter += 1
                        if ei != ej:
                            source[counter] = ej
                            target[counter] = ei
                            weights[counter] = 1
                            counter += 1
                    else:
                        source[counter] = ei
                        target[counter] = ej
                        weights[counter] = float(data[2])
                        counter += 1
                        if ei != ej:
                            source[counter] = ej
                            target[counter] = ei
                            weights[counter] = float(data[2])
                            counter += 1
                    

#            # Treat dangling nodes.
#            unique_elements = set(source)
#            unique_elements.update(set(target))
#            max_id = max(unique_elements)
#            
#            self._dangling = list(set(range(max_id)) - unique_elements)
#            
#            unique_elements = list(unique_elements)
#            
#            if len(self._dangling) > 0:
#                print('The following nodes have no outgoing edges:',self._dangling,'\n')
#                print('These nodes are stored in the your_graph_object._dangling.')
#                print('To avoid numerical difficulties we connect each dangling node to another randomly chosen node.')
#            
#            source_dangling = np.zeros(len(self._dangling)*2,int)
#            target_dangling = np.zeros(len(self._dangling)*2,int)
#            weights_dangling = np.zeros(len(self._dangling)*2,float)
#            
#            counter = 0
#            
#            min_weight = min(weights)
#            
#            for i in self._dangling:
#                j = np.random.choice(unique_elements)
#                while j == i:
#                    j = np.random.choice(unique_elements)
#                source_dangling[counter] = j
#                target_dangling[counter] = i
#                weights_dangling[counter] = min_weight
#                counter += 1
#                source_dangling[counter] = i
#                target_dangling[counter] = j
#                weights_dangling[counter] = min_weight
#                counter += 1
#                
#            # Update edges with info from dangling nodes.
#            source = np.append(source,source_dangling) 
#            target = np.append(target,target_dangling) 
#            weights = np.append(weights,weights_dangling) 
                
            if len(source) != len(target):
                print('The edgelist input is corrupted')

            self._num_edges = len(source)
            #m = self._num_edges
            #self._num_vertices = max([max(second_column),max(first_column)]) + 1
            #n = self._num_vertices

            #self.adjacency_matrix = sp.coo_matrix((np.ones(m),(first_column,second_column)), shape=(n,n))
            #self.adjacency_matrix = self.adjacency_matrix.tocsr()
            #self.adjacency_matrix = self.adjacency_matrix + self.adjacency_matrix.T
            
            #unique_elements = set(first_column).copy()
            #unique_elements.update(set(second_column))
            #self._num_vertices = len(unique_elements)
            #n = self._num_vertices
            
            #self.adjacency_matrix = self.adjacency_matrix.tocsr()[list(unique_elements), :].tocsc()[:, list(unique_elements)]
            
            self.adjacency_matrix = self.list_to_CSR(source,target,weights)
            
            self._num_vertices = self.adjacency_matrix.shape[0]
            
        elif file_type == 'gml':
            warnings.warn("Loading a gml is not efficient, we suggest using an edgelist format for this API.")
            G = nx.read_gml(filename)
            self.adjacency_matrix = nx.adjacency_matrix(G).astype(np.float64)
            self._num_edges = nx.number_of_edges(G)
            self._num_vertices = nx.number_of_nodes(G)
        elif file_type == 'graphml':
            warnings.warn("Loading a graphml is not efficient, we suggest using an edgelist format for this API.")
            G = nx.read_graphml(filename)
            self.adjacency_matrix = nx.adjacency_matrix(G).astype(np.float64)
            self._num_edges = nx.number_of_edges(G)
            self._num_vertices = nx.number_of_nodes(G)
        else:
            print('This file type is not supported')
            return
        
        self.compute_statistics()
        
    def discard_weights(self):
        """ Discard any weights that were loaded from the data file.
        This sets all the weights associated with each edge to 1.0, 
        which is our "no weight" case."""
        self.adjacency_matrix.data.fill(1.0)
        self.compute_statistics()

    def compute_statistics(self):
        """
        Computes statistics for the graph. It updates the class attributes. The user needs to read the graph first before calling
        this method by calling the read_graph method from this class.
        """
        self.d = np.ravel(self.adjacency_matrix.sum(axis=1))            
        self.dn = np.zeros(self._num_vertices)
        for i in range(self._num_vertices):
            if self.d[i] != 0:
                self.dn[i] = 1.0/self.d[i]
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

        cond_R = cut_R/min(vol_R,self.vol_G - vol_R)
        
        return cond_R

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
            return g_copy
