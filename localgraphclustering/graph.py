"""
    DESCRIPTION
    -----------

    Graph class. It implements graph loading from an edgelist or gml and provides functions that operate on the graph.

    Call help(graph.__init__) to get the documentation for the variables and functions of this class.
    Call help(graph.name_of_function) to get the documentation for function name_of_function.

    CLASS VARIABLES 
    ---------------

    1) A: sparse row format matirx
       Adjacency matrix

    2) d: floating point numpy vector 
       Degrees vector

    3) dn: floating point numpy vector 
        Component-wise reciprocal of degrees vector

    4) d_sqrt: floating point numpy vector
            Component-wise square root of degrees vector

    5) dn_sqrt: floating point numpy vector 
             Component-wise reciprocal of sqaure root degrees vector

    6) vol_G: floating point scalar 
           Volume of graph

    7) dangling: interger numpy array
              Nodes with zero edges

    8) components: list of sets 
                Each set contains the indices of a connected component of the graph

    9) number_of_components: integer
                          Number of connected components of the graph

    10) bicomponents: list of sets  
                  Each set contains the indices of a biconnected component of the graph

    11) number_of_bicomponents: integer
                                       Number of connected components of the graph

    12) conductance_vs_vol: list of dictionaries
                        The length of the list is the number of connected components of the given graph.
                        Each element of the list is a dictionary where keys are volumes of clusters and 
                        the values are conductance. It can be used to plot the conductance vs volume NCP.

    13) conductance_vs_size: list of dictionaries
                         The length of the list is the number of connected components of the given graph.
                         Each element of the list is a dictionary where keys are sizes of clusters and 
                         the values are conductance. It can be used to plot the conductance vs volume NCP.    

    14) component_conductance_vs_vol: dictionary
                                  A dictionary where keys are volumes of clusters and 
                                  the values are conductance. It can be used to plot the conductance vs volume NCP.

    15) component_conductance_vs_size: dictionary
                                   A dictionary where keys are sizes of clusters and 
                                   the values are conductance. It can be used to plot the conductance vs volume NCP.

    16) biconnected_conductance_vs_vol: List of dictionaries
                                    The length of the list is the number of connected components of the given graph.
                                    Each element of the list is a dictionary where keys are volumes of clusters and 
                                    the values are conductance. It can be used to plot the conductance vs volume NCP.

    17) biconnected_conductance_vs_size: List of dictionaries
                                     The length of the list is the number of connected components of the given graph.
                                     Each element of the list is a dictionary where keys are sizes of clusters and 
                                     the values are conductance. It can be used to plot the conductance vs volume NCP.

    18) core_numbers: dictionary
                  Core number for each vertex

    FUNCTIONS
    ---------

    1) compute_statistics(self)

    2) read_graph(self,filename,separator)

    3) is_disconnected(self)

    4) connected_components(self)

    5) biconnected_components(self)

    6) core_number(self)

    7) ncp(self, ratio, algorithm = 'fista', epsilon = 1.0e-8, max_iter = 1000, max_time_ncp = 1000, max_time_algorithm = 10)

    8) biconnected_ncp(self, ratio, algorithm = 'fista', epsilon = 1.0e-8, max_iter = 1000, max_time_ncp = 1000, max_time_algorithm = 10)

    10) ncp_one_component(self, ratio, nodes_of_component, algorithm = 'fista', epsilon = 1.0e-8, max_iter = 1000, max_time_ncp = 1000, max_time_algorithm = 10)

    11) plot_ncp(self)

    12) plot_biconnected_ncp(self)

    13) plot_one_component_ncp(self)
"""     

from scipy import sparse as sp
import scipy.sparse.linalg as splinalg
import numpy as np
import csv
import networkx as nx
import matplotlib.pylab as plt

from localgraphclustering.localCluster import *
from localgraphclustering.eig2L_subgraph import eig2L_subgraph
import time

def import_text(filename, separator):
    for line in csv.reader(open(filename), delimiter=separator, 
                           skipinitialspace=True):
        if line:
            yield line

class graph:
    def __init__(self):
        """
            CLASS VARIABLES
            ---------------

            1) A: sparse row format matirx
                  Adjacency matrix

            2) d: floating point numpy vector 
                  Degrees vector

            3) dn: floating point numpy vector 
                   Component-wise reciprocal of degrees vector

            4) d_sqrt: floating point numpy vector
                       Component-wise square root of degrees vector

            5) dn_sqrt: floating point numpy vector 
                        Component-wise reciprocal of sqaure root degrees vector

            6) vol_G: floating point scalar 
                      Volume of graph

            7) dangling: interger numpy array
                         Nodes with zero edges

            8) components: list of sets 
                           Each set contains the indices of a connected component of the graph

            9) number_of_components: integer
                                     Number of connected components of the graph

            10) bicomponents: list of sets  
                              Each set contains the indices of a biconnected component of the graph

            11) number_of_bicomponents: integer
                                        Number of connected components of the graph

            12) conductance_vs_vol: list of dictionaries
                                    The length of the list is the number of connected components of the given graph.
                                    Each element of the list is a dictionary where keys are volumes of clusters and 
                                    the values are conductance. It can be used to plot the conductance vs volume NCP.

            13) conductance_vs_size: list of dictionaries
                                     The length of the list is the number of connected components of the given graph.
                                     Each element of the list is a dictionary where keys are sizes of clusters and 
                                     the values are conductance. It can be used to plot the conductance vs volume NCP.    

            14) component_conductance_vs_vol: dictionary
                                              A dictionary where keys are volumes of clusters and 
                                              the values are conductance. It can be used to plot the conductance vs volume NCP.

            15) component_conductance_vs_size: dictionary
                                               A dictionary where keys are sizes of clusters and 
                                               the values are conductance. It can be used to plot the conductance vs volume NCP.

            16) biconnected_conductance_vs_vol: List of dictionaries
                                                The length of the list is the number of connected components of the given graph.
                                                Each element of the list is a dictionary where keys are volumes of clusters and 
                                                the values are conductance. It can be used to plot the conductance vs volume NCP.

            17) biconnected_conductance_vs_size: List of dictionaries
                                                 The length of the list is the number of connected components of the given graph.
                                                 Each element of the list is a dictionary where keys are sizes of clusters and 
                                                 the values are conductance. It can be used to plot the conductance vs volume NCP.

            18) core_numbers: dictionary
                              Core number for each vertex
        """
        self.A = []
        self.d = []
        self.dn = []
        self.d_sqrt = []
        self.dn_sqrt = []
        self.vol_G = []
        self.dangling = []
        self.components = []
        self.number_of_components = []
        self.number_of_bicomponents = []
        self.bicomponents = []
        self.conductance_vs_vol = []
        self.conductance_vs_size = []
        self.component_conductance_vs_vol = []
        self.component_conductance_vs_size = []
        self.biconnected_conductance_vs_vol = []
        self.biconnected_conductance_vs_size = []
        self.core_numbers = []
        
    def compute_statistics(self):
        """
            DESCRIPTION
            -----------

            Computes statistics for the graph. It stores the results in class variables 
            d, dangling, dn, d_sqrt, dn_sqrt and vol_G. The user needs to read the graph first before calling
            this function by calling the read_graph function from this class.

            Call help(graph.__init__) to get the documentation for the variables of this class.
            

            RETURNS
            -------
           
            dangling: interger numpy array
                      Nodes with zero edges.
            
            d: floating point numpy vector 
               Degrees vector
           
            dn: floating point numpy vector 
                Component-wise reciprocal of degrees vector
            
            d_sqrt: floating point numpy vector
                    Component-wise square root of degrees vector
           
            dn_sqrt: floating point numpy vector 
                     Component-wise reciprocal of sqaure root degrees vector
            
            vol_G: floating point scalar 
                   Volume of graph
        """
        n = self.A.shape[0]
        
        self.d = np.ravel(self.A.sum(axis=1))
        self.dangling = np.where(self.d == 0)[0]
        if self.dangling.shape[0] > 0:
            print('The following nodes have no outgoing edges:',self.dangling,'\n')
            print('These nodes are stored in the your_graph_object.dangling.')
            print('To avoid numerical difficulties we connect each dangling node to another randomly chosen node.')
            
            self.A = sp.lil_matrix(self.A)
            
            for i in self.dangling:
                numbers = list(range(0,i))+list(range(i + 1,n - 1))
                j = np.random.choice(numbers)
                self.A[i,j] = 1
                self.A[j,i] = 1
                
            self.A = sp.csr_matrix(self.A)

            self.d = np.ravel(self.A.sum(axis=1))
        
        self.dn = 1.0/self.d
        self.d_sqrt = np.sqrt(self.d)
        self.dn_sqrt = np.sqrt(self.dn)
        self.vol_G = np.sum(self.d)

    def read_graph(self,filename,file_type = 'edgelist',separator = '\t'):
        """
            DESCRIPTION
            -----------

            Reads the graph from an edgelist or gml and initializes the adjecancy matrix which is stored in class variable A.

            Call help(graph.__init__) to get the documentation for the variables of this class.
            
            PARAMETERS
            ----------
            
            filename: string, name of the file, for example 'JohnsHopkins.edgelist' or 'JohnsHopkins.gml'.
            
            file_type: string, type of file. Currently only 'edgelist' and 'gml' are supported.
                       Default = 'edgelist'
                       
            separator: string, used if file_type = 'edgelist'
                       Default = '\t'
            
            RETURNS
            -------
           
            The output can be accessed from the graph object that calls this function.
           
            A: sparse row format matirx
               Adjacency matrix
        """
        
        if file_type == 'edgelist':
        
            first_column = []
            second_column = []

            for data in import_text(filename, separator):
                first_column.extend([int(data[0])])
                second_column.extend([int(data[1])])

            if len(first_column) != len(second_column):
                print('The edgelist input is corrupted')

            m = len(first_column)
            n = max([max(second_column),max(first_column)]) + 1

            self.A = sp.coo_matrix((np.ones(m),(first_column,second_column)), shape=(n,n))
            self.A = self.A.tocsr()
            self.A = self.A + self.A.T
        elif file_type == 'gml':
            G = nx.read_gml(filename)
            self.A = nx.adjacency_matrix(G).astype(np.float64)
        else:
            print('This file type is not supported')
            return
            
        self.compute_statistics()
        
    def connected_components(self):
        """
            DESCRIPTION
            -----------

            Computes the connected components of the graph. It stores the results in class variables components 
            and number_of_components. The user needs to call read the graph 
            first before calling this function by calling the read_graph function from this class.

            Call help(graph.__init__) to get the documentation for the variables of this class.
            
            RETURNS
            -------
           
            The output can be accessed from the graph object that calls this function.
           
            components: list of sets 
                        Each set contains the indices of a connected component of the graph

            number_of_components: integer
                                  Number of connected components of the graph
        """
        
        g_nx = nx.from_scipy_sparse_matrix(self.A)

        self.components = list(nx.connected_components(g_nx))

        self.number_of_components = nx.number_connected_components(g_nx)
        
        print('There are ', self.number_of_components, ' connected components in the graph')        
        
    def is_disconnected(self):
        """
            DESCRIPTION
            -----------
            
            The output can be accessed from the graph object that calls this function.

            Checks if the graph is a disconnected graph. It prints the result as a comment and 
            returns True if the graph is disconnected, or false otherwise. The user needs to 
            call read the graph first before calling this function by calling the read_graph function from this class.
            
            RETURNS
            -------
           
            True: if connected 

            False: if disconnected
        """
        
        if self.d == []:
            print('The graph has to be read first.')
            return
    
        #n = self.A.shape[0]

        #D_sqrt_neg = sp.spdiags(self.dn_sqrt.transpose(), 0, n, n)

        #L = sp.identity(n) - D_sqrt_neg.dot((self.A.dot(D_sqrt_neg)))

        #emb_eig_val, p = splinalg.eigs(L, which='SM', k=2)
        
        self.connected_components()
        
        if self.number_of_components > 1:
            print('The graph is a disconnected graph.')
            return True
        else: 
            print('The graph is not a disconnected graph.')
            return False
        
    def biconnected_components(self):
        """
            DESCRIPTION
            -----------
        
            Computes the biconnected components of the graph. It stores the results in class variables bicomponents 
            and number_of_bicomponents. The user needs to call read the graph first before calling this 
            function by calling the read_graph function from this class.
        
            Call help(graph.__init__) to get the documentation for the variables of this class.
            
            RETURNS
            -------
           
            The output can be accessed from the graph object that calls this function.
           
            bicomponents: list of sets  
                          Each set contains the indices of a biconnected component of the graph

            number_of_bicomponents: integer
                                    Number of biconnected components of the graph
        """
        
        g_nx = nx.from_scipy_sparse_matrix(self.A)

        self.bicomponents = list(nx.biconnected_components(g_nx))
        
        self.number_of_bicomponents = len(self.bicomponents)
        
    def core_number(self):
        """
            DESCRIPTION
            -----------
        
            From Networkx: Return the core number for each vertex. A k-core is a maximal 
            subgraph that contains nodes of degree k or more. The core number of a node 
            is the largest value k of a k-core containing that node. The user needs to 
            call read the graph first before calling this function by calling the read_graph 
            function from this class. The output can be accessed from the graph object that 
            calls this function. It stores the results in class variables core_numbers.
        
            Call help(graph.__init__) to get the documentation for the variables of this class.
            
            RETURNS
            -------
           
            The output can be accessed from the graph object that calls this function.
           
            core_numbers: list of integers  
                          Each set contains the indices of a biconnected component of the graph
        """
        
        g_nx = nx.from_scipy_sparse_matrix(self.A)

        self.core_numbers = nx.core_number(g_nx)

    def ncp(self, ratio, algorithm = 'fista', epsilon = 1.0e-4, max_iter = 1000, max_time_ncp = 1000, max_time_algorithm = 10, cpp = True):
        """
           DESCRIPTION
           -----------

           Network Community Profile for all connected components of the graph. For details please refer to: 
           Jure Leskovec, Kevin J Lang, Anirban Dasgupta, Michael W Mahoney. Community structure in 
           large networks: Natural cluster sizes and the absence of large well-defined clusters.
           The NCP is computed for each connected component of the given graph.

           PARAMETERS (mandatory)
           ---------------------     

           ratio:  float, double
                   Ratio of nodes to be used for computation of NCP.
                   It should be between 0 and 1.

           PARAMETERS (optional)
           ---------------------

           algorithm: string
                      default == 'fista'
                      Algorithm for spectral local graph clustering. 
                      Options: 'fista', 'ista', 'acl'.

           epsilon: float, double
                    default = 1.0e-4
                    Termination tolerance for l1-regularized PageRank, i.e., applies to all algorithms.

           max_iter: integer
                     default = 10000
                     Maximum number of iterations of FISTA, ISTA or ACL.

           max_time_ncp: float, double
                         default = 1000
                         Maximum time in seconds for NCP calculation.

           max_time_algorithm: float, double
                               default = 10
                               Maximum time in seconds for each algorithm run during the NCP calculation.
                               
           cpp: boolean
                Use the faster C++ version of FISTA or not.

           RETURNS
           -------

           The output can be accessed from graph object that calls this function.

           conductance_vs_vol: a list of dictionaries
                               The length of the list is the number of connected components of the given graph.
                               Each element of the list is a dictionary where keys are volumes of clusters and 
                               the values are conductance. It can be used to plot the conductance vs volume NCP.

           conductance_vs_size: a list of dictionaries
                                The length of the list is the number of connected components of the given graph.
                                Each element of the list is a dictionary where keys are sizes of clusters and 
                                the values are conductance. It can be used to plot the conductance vs volume NCP.

        """         
        if ratio < 0 or ratio > 1:
            print("Ratio must be between 0 and 1.")
            return []        
        
        self.connected_components()
        
        number_of_components = self.number_of_components
        
        if number_of_components <= 0:
            print("There are no connected components in the given graph")
            return
        
        self.conductance_vs_vol = []
        self.conductance_vs_size = []
        for i in range(number_of_components):
            self.conductance_vs_vol.append({})
            self.conductance_vs_size.append({})
        
        start = time.time()        
        
        for cmp in range(number_of_components):
            
            nodes_of_component = list(self.components[cmp])
            g_copy = graph()
            g_copy.A = self.A[nodes_of_component,:].tocsc()[:,nodes_of_component].tocsr()
            g_copy.compute_statistics()          
        
            n = g_copy.A.shape[0]

            n_nodes = min(np.ceil(ratio*n),n)
            n_nodes = int(n_nodes)

            nodes = np.random.choice(np.arange(0,n), n_nodes, replace=False)

            lc = localCluster()

            for node in nodes:

#                if algorithm == 'fista':
#                    p = fista_dinput_dense(node, g_copy, alpha = 0.15, rho = rho, epsilon = epsilon, max_iter = 10, max_time = max_time_algorithm)
#                elif algorithm == 'ista':
#                    p = ista_dinput_dense(node, g_copy, alpha = 0.15, rho = rho, epsilon = epsilon, max_iter = 10, max_time = max_time_algorithm)
#                elif algorithm == 'acl':
#                    p = acl_list(node, g_copy, alpha = 0.15, rho = rho, max_iter = 100, max_time = max_time_algorithm)
#                else:
#                    print("There is no such algorithm provided")
#                    return

#                rr = p.nonzero()[0]

#                eigv, lambda_val = eig2L_subgraph(g_copy.A, rr)
#                lambda_val = np.real(lambda_val)
#                step = (2*lambda_val - lambda_val/2)/4
#                if lambda_val == 0 or step == 0:
#                    a_list = np.arange(1.0e-2,0.9,0.2)
#                else:
#                    a_list = np.arange(lambda_val/2,2*lambda_val,step)
    
                rho_list = [1.0e-14,1.0e-12,1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4]
        
                for rho in rho_list:
                
                    #a_list = np.arange(1.0e-2,0.999,0.09)
                    a_list = [1-0.99]

                    for alpha in a_list:

                        if algorithm == 'fista':
                            if not cpp:
                                lc.fista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm, cpp = cpp)
                            else:
                                lc.fista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm, cpp = cpp) 

                            S_l1pr = lc.best_cluster_fista
                            cond = lc.best_conductance_fista
                        elif algorithm == 'ista':
                            lc.ista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm)              
                            S_l1pr = lc.best_cluster_ista
                            cond = lc.best_conductance_ista
                        elif algorithm == 'acl':
                            lc.acl(node, g_copy, alpha = alpha, rho = rho, max_iter = max_iter, max_time = max_time_algorithm)
                            S_l1pr = lc.best_cluster_acl
                            cond = lc.best_conductance_acl
                        else:
                            print("There is no such algorithm provided")
                            return

                        vol = sum(g_copy.d[S_l1pr])
                        size = len(S_l1pr)

                        if vol in self.conductance_vs_vol[cmp]:
                            if cond <= self.conductance_vs_vol[cmp][vol]:
                                self.conductance_vs_vol[cmp][vol] = cond
                        else:
                            self.conductance_vs_vol[cmp][vol] = cond  

                        if size in self.conductance_vs_size[cmp]:
                            if cond <= self.conductance_vs_size[cmp][size]:
                                self.conductance_vs_size[cmp][size] = cond
                        else:
                            self.conductance_vs_size[cmp][size] = cond     

                        end = time.time()
                        
                if end - start > max_time_ncp:
                    break

    def biconnected_ncp(self, ratio, algorithm = 'fista', epsilon = 1.0e-4, max_iter = 1000, max_time_ncp = 1000, max_time_algorithm = 10, cpp = True):
        """
           DESCRIPTION
           -----------

           Network Community Profile for all biconnected components of the graph. For details please refer to: 
           Jure Leskovec, Kevin J Lang, Anirban Dasgupta, Michael W Mahoney. Community structure in 
           large networks: Natural cluster sizes and the absence of large well-defined clusters.

           PARAMETERS (mandatory)
           ----------------------    

           ratio:  float, double
                   Ratio of nodes to be used for computation of NCP.
                   It should be between 0 and 1.

           PARAMETERS (optional)
           ---------------------

           algorithm: string
                      default == 'fista'
                      Algorithm for spectral local graph clustering. 
                      Options: 'fista', 'ista', 'acl'.

           epsilon: float, double
                    default = 1.0e-4
                    Termination tolerance for l1-regularized PageRank, i.e., applies to FISTA and ISTA algorithms

           max_iter: integer
                     default = 10000
                     Maximum number of iterations of FISTA, ISTA or ACL.

           max_time_ncp: float, double
                         default = 1000
                         Maximum time in seconds for NCP calculation.

           max_time_algorithm: float, double
                               default = 10
                               Maximum time in seconds for each algorithm run during the NCP calculation.
                               
           cpp: boolean
                Use the faster C++ version of FISTA or not.

           RETURNS
           -------
           
           The output can be accessed from graph object that calls this function.

           biconnected_conductance_vs_vol: a list of dictionaries
                               The length of the list is the number of connected components of the given graph.
                               Each element of the list is a dictionary where keys are volumes of clusters and 
                               the values are conductance. It can be used to plot the conductance vs volume NCP.

           biconnected_conductance_vs_size: a list of dictionaries
                               The length of the list is the number of connected components of the given graph.
                               Each element of the list is a dictionary where keys are sizes of clusters and 
                               the values are conductance. It can be used to plot the conductance vs volume NCP.

        """         
        if ratio < 0 or ratio > 1:
            print("Ratio must be between 0 and 1.")
            return []        
        
        self.biconnected_components()
        
        number_of_components = self.number_of_bicomponents
        
        if number_of_components <= 0:
            print("There are no biconnected components in the given graph")
            return
        
        self.biconnected_conductance_vs_vol = []
        self.biconnected_conductance_vs_size = []
        for i in range(number_of_components):
            self.biconnected_conductance_vs_vol.append({})
            self.biconnected_conductance_vs_size.append({})

        start = time.time()  
        
        for cmp in range(number_of_components):
            
            nodes_of_component = list(self.bicomponents[cmp])
            g_copy = graph()
            g_copy.A = self.A[nodes_of_component,:].tocsc()[:,nodes_of_component].tocsr()
            g_copy.compute_statistics()          
        
            n = g_copy.A.shape[0]

            n_nodes = min(np.ceil(ratio*n),n)
            n_nodes = int(n_nodes)

            nodes = np.random.choice(np.arange(0,n), n_nodes, replace=False)
            
            lc = localCluster()

            for node in nodes:

#                if algorithm == 'fista':
#                    p = fista_dinput_dense(node, g_copy, alpha = 0.15, rho = rho, epsilon = epsilon, max_iter = 10, max_time = max_time_algorithm)
#                elif algorithm == 'ista':
#                    p = ista_dinput_dense(node, g_copy, alpha = 0.15, rho = rho, epsilon = epsilon, max_iter = 10, max_time = max_time_algorithm)
#                elif algorithm == 'acl':
#                    p = acl_list(node, g_copy, alpha = 0.15, rho = rho, max_iter = 100, max_time = max_time_algorithm)
#                else:
#                    print("There is no such algorithm provided")
#                    return

#                rr = p.nonzero()[0]

#                eigv, lambda_val = eig2L_subgraph(g_copy.A, rr)
#                lambda_val = np.real(lambda_val)
#                step = (2*lambda_val - lambda_val/2)/4
#                if lambda_val == 0 or step == 0:
#                    a_list = np.arange(1.0e-2,0.9,0.2)
#                else:
#                    a_list = np.arange(lambda_val/2,2*lambda_val,step)

                rho_list = [1.0e-14,1.0e-12,1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4]
        
                for rho in rho_list:
                
                    #a_list = np.arange(1.0e-2,0.999,0.09)
                    a_list = [1-0.99]

                    for alpha in a_list:

                        if algorithm == 'fista':
                            if not cpp:
                                lc.fista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm, cpp = cpp)
                            else:
                                lc.fista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm, cpp = cpp) 

                            S_l1pr = lc.best_cluster_fista
                            cond = lc.best_conductance_fista
                        elif algorithm == 'ista':
                            lc.ista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm)  
                            S_l1pr = lc.best_cluster_ista
                            cond = lc.best_conductance_ista
                        elif algorithm == 'acl':
                            lc.acl(node, g_copy, alpha = alpha, rho = rho, max_iter = max_iter, max_time = max_time_algorithm)
                            S_l1pr = lc.best_cluster_acl
                            cond = lc.best_conductance_acl
                        else:
                            print("There is no such algorithm provided")
                            return

                        vol = sum(g_copy.d[S_l1pr])
                        size = len(S_l1pr)

                        if vol in self.biconnected_conductance_vs_vol[cmp]:
                            if cond <= self.biconnected_conductance_vs_vol[cmp][vol]:
                                self.biconnected_conductance_vs_vol[cmp][vol] = cond
                        else:
                            self.biconnected_conductance_vs_vol[cmp][vol] = cond  

                        if size in self.biconnected_conductance_vs_size[cmp]:
                            if cond <= self.biconnected_conductance_vs_size[cmp][size]:
                                self.biconnected_conductance_vs_size[cmp][size] = cond
                        else:
                            self.biconnected_conductance_vs_size[cmp][size] = cond     

                        end = time.time()

                if end - start > max_time_ncp:
                    break    
                        
    def ncp_one_component(self, ratio, nodes_of_component, algorithm = 'fista', epsilon = 1.0e-4, max_iter = 1000, max_time_ncp = 1000, max_time_algorithm = 10, cpp = True):
        """
           DESCRIPTION
           -----------

           Network Community Profile for a given set of nodes. For details please refer to: 
           Jure Leskovec, Kevin J Lang, Anirban Dasgupta, Michael W Mahoney. Community structure in 
           large networks: Natural cluster sizes and the absence of large well-defined clusters.
           This function checks if the given nodes are connected.

           PARAMETER (mandatory)
           ---------------------     

           ratio:  float, double
                   Ratio of nodes to be used for computation of NCP.
                   It should be between 0 and 1.
                   
           nodes_of_component: list
                               A list of nodes.

           PARAMETERS (optional)
           ---------------------

           algorithm: string
                      default == 'fista'
                      Algorithm for spectral local graph clustering. 
                      Options: 'fista', 'ista', 'acl'.

           epsilon: float, double
                    default = 1.0e-4
                    Termination tolerance for l1-regularized PageRank, i.e., applies to FISTA and ISTA algorithms

           max_iter: integer
                     default = 10000
                     Maximum number of iterations of FISTA, ISTA or ACL.

           max_time_ncp: float, double
                         default = 1000
                         Maximum time in seconds for NCP calculation.

           max_time_algorithm: float, double
                               default = 10
                               Maximum time in seconds for each algorithm run during the NCP calculation.
                              
           cpp: boolean
                Use the faster C++ version of FISTA or not.

           RETURNS
           -------

           The output can be accessed from graph object that calls this function.

           component_conductance_vs_vol: dictionary
                               A dictionary where keys are volumes of clusters and 
                               the values are conductance. It can be used to plot the conductance vs volume NCP.

           component_conductance_vs_size: dictionary
                               A dictionary where keys are sizes of clusters and 
                               the values are conductance. It can be used to plot the conductance vs volume NCP.

        """         
        if ratio < 0 or ratio > 1:
            print("Ratio must be between 0 and 1.")
            return []       
        
        nodes_of_component = list(nodes_of_component)
        
        g_nx = nx.from_scipy_sparse_matrix(self.A[nodes_of_component,:].tocsc()[:,nodes_of_component].tocsr())
        is_connected = nx.is_connected(g_nx)
        if not is_connected: 
            print("The given nodes_of_component are not connected")
        
        self.component_conductance_vs_vol = {}
        self.component_conductance_vs_size = {}
            
        g_copy = graph()
        g_copy.A = self.A[nodes_of_component,:].tocsc()[:,nodes_of_component].tocsr()
        g_copy.compute_statistics()          
        
        n = g_copy.A.shape[0]

        n_nodes = min(np.ceil(ratio*n),n)
        n_nodes = int(n_nodes)

        nodes = np.random.choice(np.arange(0,n), n_nodes, replace=False)
        
        lc = localCluster()

        start = time.time()

        for node in nodes:

#            if algorithm == 'fista':
#                p = fista_dinput_dense(node, g_copy, alpha = 0.15, rho = rho, epsilon = epsilon, max_iter = 10, max_time = max_time_algorithm)
#            elif algorithm == 'ista':
#                p = ista_dinput_dense(node, g_copy, alpha = 0.15, rho = rho, epsilon = epsilon, max_iter = 10, max_time = max_time_algorithm)
#            elif algorithm == 'acl':
#                p = acl_list(node, g_copy, alpha = 0.15, rho = rho, max_iter = 100, max_time = max_time_algorithm)
#            else:
#                print("There is no such algorithm provided")
#                return

#            rr = p.nonzero()[0]

#            eigv, lambda_val = eig2L_subgraph(g_copy.A, rr)
#            lambda_val = np.real(lambda_val)
#            step = (2*lambda_val - lambda_val/2)/4
#            if lambda_val == 0 or step == 0:
#                a_list = np.arange(1.0e-2,0.9,0.2)
#            else:
#                a_list = np.arange(lambda_val/2,2*lambda_val,step)

            rho_list = [1.0e-14,1.0e-12,1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4]
        
            for rho in rho_list:

                #a_list = np.arange(1.0e-2,0.999,0.09)
                a_list = [1-0.99]

                for alpha in a_list:

                    if algorithm == 'fista':
                            if not cpp:
                                lc.fista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm, cpp = cpp)
                            else:
                                lc.fista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm, cpp = cpp)

                            S_l1pr = lc.best_cluster_fista
                            cond = lc.best_conductance_fista
                    elif algorithm == 'ista':
                        lc.ista(node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time_algorithm)  
                        S_l1pr = lc.best_cluster_ista
                        cond = lc.best_conductance_ista
                    elif algorithm == 'acl':
                        lc.acl(node, g_copy, alpha = alpha, rho = rho, max_iter = max_iter, max_time = max_time_algorithm)
                        S_l1pr = lc.best_cluster_acl
                        cond = lc.best_conductance_acl
                    else:
                        print("There is no such algorithm provided")
                        return

                    vol = sum(g_copy.d[S_l1pr])
                    size = len(S_l1pr)

                    if vol in self.component_conductance_vs_vol:
                        if cond <= self.component_conductance_vs_vol[vol]:
                            self.component_conductance_vs_vol[vol] = cond
                    else:
                        self.component_conductance_vs_vol[vol] = cond  

                    if size in self.component_conductance_vs_size:
                        if cond <= self.component_conductance_vs_size[size]:
                            self.component_conductance_vs_size[size] = cond
                    else:
                        self.component_conductance_vs_size[size] = cond     

                    end = time.time()

            if end - start > max_time_ncp:
                break
                
    def plot_ncp(self):
        """
            DESCRIPTION
            -----------

            Plots the Network Community Profile for each connected component of the graph.
        """
        if self.conductance_vs_size == []:
            print("graph.ncp(ratio) has to be run first.")
            return
            
        
        for i in range(self.number_of_components):
        
            lists = sorted(self.conductance_vs_vol[i].items())
            x, y = zip(*lists)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            plt.plot(x, y)
            
            ax.set_xlabel('Volume')
            ax.set_ylabel('Minimum conductance')  
            
            ax.set_title('Min. Conductance vs. Volume NCP for component ' + str(i+1))
            
            plt.show()

            lists = sorted(self.conductance_vs_size[i].items())
            x, y = zip(*lists)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            plt.plot(x, y)
            
            ax.set_xlabel('Size')
            ax.set_ylabel('Minimum conductance')  
            
            ax.set_title('Min. Conductance vs. Size NCP for component ' + str(i+1))
            
            plt.show()  
            
    def plot_biconnected_ncp(self):
        """
            DESCRIPTION
            -----------

            Plots the Network Community Profile for each biconnected component of the graph.
        """
        if self.biconnected_conductance_vs_vol == []:
            print("graph.biconnected_ncp(ratio) has to be run first.")
            return
            
        
        for i in range(self.number_of_bicomponents):
        
            lists = sorted(self.biconnected_conductance_vs_vol[i].items())
            x, y = zip(*lists)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            plt.plot(x, y)
            
            ax.set_xlabel('Volume')
            ax.set_ylabel('Minimum conductance')  
            
            ax.set_title('Min. Conductance vs. Volume NCP for biconnected component ' + str(i+1))
            
            plt.show()

            lists = sorted(self.biconnected_conductance_vs_size[i].items())
            x, y = zip(*lists)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            plt.plot(x, y)
            
            ax.set_xlabel('Size')
            ax.set_ylabel('Minimum conductance')  
            
            ax.set_title('Min. Conductance vs. Size NCP for biconnected component ' + str(i+1))
            
            plt.show()    
            
    def plot_one_component_ncp(self):
        """
            DESCRIPTION
            -----------

            Plots the Network Community Profile for the given connected component.
        """
        if self.component_conductance_vs_vol == []:
            print("graph.ncp_one_component(ratio,nodes_of_component) has to be run first.")
            return
        
        lists = sorted(self.component_conductance_vs_vol.items())
        x, y = zip(*lists)

        fig = plt.figure()
        ax = fig.add_subplot(111)
            
        plt.plot(x, y)
            
        ax.set_xlabel('Volume')
        ax.set_ylabel('Minimum conductance')  
            
        ax.set_title('Min. Conductance vs. Volume NCP for given component')
            
        plt.show()

        lists = sorted(self.component_conductance_vs_size.items())
        x, y = zip(*lists)

        fig = plt.figure()
        ax = fig.add_subplot(111)
            
        plt.plot(x, y)
            
        ax.set_xlabel('Size')
        ax.set_ylabel('Minimum conductance')  
            
        ax.set_title('Min. Conductance vs. Size NCP for given component')
            
        plt.show()              
        
