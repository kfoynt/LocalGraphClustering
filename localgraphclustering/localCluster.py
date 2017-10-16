"""
    DESCRIPTION
    -----------

    LocalCluster class. It implements local graph clustering methods.

    Call help(localCluster.__init__) to get the documentation for the variables of this class.
    Call help(localCluster.name_of_function) to get the documentation for function name_of_function.

    CLASS VARIABLES 
    ---------------

    1) node_embedding_acl: numpy array, float
                           1D node embedding for each node of the graph when using ACL function.
       
    2) best_cluster_acl: list
                         A list of nodes that correspond to the cluster with the best 
                         conductance that was found by ACL function.
                         
    3) best_conductance_acl: float
                             Conductance value that corresponds to the cluster with the best 
                             conductance that was found by ACL function.
                             
    4) sweep_profile_acl: list of objects
                          A two dimensional list of objects. For example,
                          sweep_profile[0] contains a numpy array with all conductances for all
                          clusters that were calculated by sweep_cut.
                          sweep_profile[1] is a multidimensional list that contains the indices
                          of all clusters that were calculated by sweep_cut. For example,
                          sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                          that was calculated by sweep_cut. The set of indices in sweep_profile[1][5] also correspond 
                          to conductance in sweep_profile[0][5]. The number of clusters is unknwon apriori 
                          and depends on the data and that parameter setting of ACL function.  
                          
    5) node_embedding_ista: numpy array, float
                            Similar to node_embedding_acl but using ISTA function.
       
    6) best_cluster_ista: list
                          Similar to best_cluster_ista but using ISTA function.
                         
    7) best_conductance_ista: float
                              Similar to best_conductance_ista but using ISTA function.
                             
    8) sweep_profile_ista: list of objects
                           Similar to sweep_profile_ista but using ISTA function. 
                           
    9) node_embedding_fista: numpy array, float
                             Similar to node_embedding_fista but using FISTA function.
       
    10) best_cluster_fista: list
                            Similar to best_cluster_fista but using FISTA function.
                         
    11) best_conductance_fista: float
                                Similar to best_conductance_fista but using FISTA function.
                             
    12) sweep_profile_fista: list of objects
                             Similar to sweep_profile_fista but using FISTA function. 
                             
    13) node_embedding_nibble: numpy array, float
                               Similar to node_embedding_nibble but using PageRank Nibble function.
        
    14) best_cluster_nibble: list
                             Similar to best_cluster_nibble but using PageRank Nibble function.
                         
    15) best_conductance_nibble: float
                                 Similar to best_conductance_nibble but using PageRank Nibble function.
                             
    16) sweep_profile_nibble: list of objects
                              Similar to sweep_profile_nibble but using PageRank Nibble function. 

    17) volume_profile_nibble: list of objects
                               A two dimensional list of objects which stores information about clusters
                               which have volume larger than the input vol and les than 2/3 of the volume
                               of the whole graph. For example, volume_profile[0] contains a list 
                               with all conductances for all clusters that were calculated by sweep_cut and 
                               also satisfy the previous volume constraint.
                               volume_profile[1] is a multidimensional list that contains the indices
                               of all clusters that were calculated by sweep_cut and also satisfy the previous 
                               volume constraint. For example, volume_profile[1][5] is a list that contains the 
                               indices of the 5th cluster that was calculated by sweep_cut and also satisfies 
                               the previous volume constraint. The set of indices in volume_profile[1][5] also correspond 
                               to conductance in volume_profile[0][5]. The number of clusters is unknwon apriori and 
                               depends on the data and that parameter setting of PageRank Nibble function. 
    FUNCTIONS
    ---------

    1) acl(ref_node, g, alpha = 0.15, rho = 1.0e-5, max_iter = 100000, max_time = 100)
    
    2) ista(ref_node, g, alpha = 0.15, rho = 1.0e-5, epsilon = 1.0e-2, max_iter = 10000, max_time = 100)
    
    3) fista(ref_node, g, alpha = 0.15, rho = 1.0e-5, epsilon = 1.0e-8, max_iter = 10000, vol_G = -1, max_time = 100, cpp = True)
   
    4) page_rank_nibble(g, ref_node, vol, phi = 0.5, algorithm = 'fista', epsilon = 1.0e-2, max_iter = 10000, max_time = 100, calculate_component = 0, cpp = True)
"""     
from localgraphclustering.acl_list import acl_list
from localgraphclustering.sweepCut import *
from localgraphclustering.fista_dinput_dense import fista_dinput_dense
from localgraphclustering.proxl1PRaccel import proxl1PRaccel
from localgraphclustering.ista_dinput_dense import ista_dinput_dense

import numpy as np
import networkx as nx
import copy as cp

class localCluster:
    
    def __init__(self):
        """
            CLASS VARIABLES 
            ---------------

            1) node_embedding_acl: numpy array, float
                                   1D node embedding for each node of the graph when using ACL function.

            2) best_cluster_acl: list
                                 A list of nodes that correspond to the cluster with the best 
                                 conductance that was found by ACL function.

            3) best_conductance_acl: float
                                     Conductance value that corresponds to the cluster with the best 
                                     conductance that was found by ACL function.

            4) sweep_profile_acl: list of objects
                                  A two dimensional list of objects. For example,
                                  sweep_profile[0] contains a numpy array with all conductances for all
                                  clusters that were calculated by sweep_cut.
                                  sweep_profile[1] is a multidimensional list that contains the indices
                                  of all clusters that were calculated by sweep_cut. For example,
                                  sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                                  that was calculated by sweep_cut. The set of indices in sweep_profile[1][5] also correspond 
                                  to conductance in sweep_profile[0][5]. The number of clusters is unknwon apriori 
                                  and depends on the data and that parameter setting of ACL function.  

            5) node_embedding_ista: numpy array, float
                                    Similar to node_embedding_acl but using ISTA function.

            6) best_cluster_ista: list
                                  Similar to best_cluster_ista but using ISTA function.

            7) best_conductance_ista: float
                                      Similar to best_conductance_ista but using ISTA function.

            8) sweep_profile_ista: list of objects
                                   Similar to sweep_profile_ista but using ISTA function. 

            9) node_embedding_fista: numpy array, float
                                     Similar to node_embedding_fista but using FISTA function.

            10) best_cluster_fista: list
                                    Similar to best_cluster_fista but using FISTA function.

            11) best_conductance_fista: float
                                        Similar to best_conductance_fista but using FISTA function.

            12) sweep_profile_fista: list of objects
                                     Similar to sweep_profile_fista but using FISTA function. 

            13) node_embedding_nibble: numpy array, float
                                       Similar to node_embedding_nibble but using PageRank Nibble function.

            14) best_cluster_nibble: list
                                     Similar to best_cluster_nibble but using PageRank Nibble function.

            15) best_conductance_nibble: float
                                         Similar to best_conductance_nibble but using PageRank Nibble function.

            16) sweep_profile_nibble: list of objects
                                      Similar to sweep_profile_nibble but using PageRank Nibble function. 

            17) volume_profile_nibble: list of objects
                                       A two dimensional list of objects which stores information about clusters
                                       which have volume larger than the input vol and les than 2/3 of the volume
                                       of the whole graph. For example, volume_profile[0] contains a list 
                                       with all conductances for all clusters that were calculated by sweep_cut and 
                                       also satisfy the previous volume constraint.
                                       volume_profile[1] is a multidimensional list that contains the indices
                                       of all clusters that were calculated by sweep_cut and also satisfy the previous 
                                       volume constraint. For example, volume_profile[1][5] is a list that contains the 
                                       indices of the 5th cluster that was calculated by sweep_cut and also satisfies 
                                       the previous volume constraint. The set of indices in volume_profile[1][5] also correspond 
                                       to conductance in volume_profile[0][5]. The number of clusters is unknwon apriori and 
                                       depends on the data and that parameter setting of PageRank Nibble function. 
        """
        self.node_embedding_acl = []
        self.best_cluster_acl = []
        self.best_conductance_acl = []
        self.sweep_profile_acl = []
        
        self.node_embedding_ista = []
        self.best_cluster_ista = []
        self.best_conductance_ista = []
        self.sweep_profile_ista = []
        
        self.node_embedding_fista = []
        self.best_cluster_fista = []
        self.best_conductance_fista = []
        self.sweep_profile_fista = []
        
        self.node_embedding_nibble = []
        self.best_cluster_nibble = []
        self.best_conductance_nibble = []
        self.sweep_profile_nibble = []
     
    def acl(self, ref_node, g, alpha = 0.15, rho = 1.0e-5, max_iter = 100000, max_time = 100):
        """
           DESCRIPTION
           -----------

           Andersen Chung and Lang (ACL) Algorithm. For details please refer to: 
           R. Andersen, F. Chung and K. Lang. Local Graph Partitioning using PageRank Vectors
           link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf

           PARAMETERS (mandatory)
           ----------------------

           ref_node:  integer
                      The reference node, i.e., node of interest around which
                      we are looking for a target cluster.

           g:         graph object

           PARAMETERS (optional)
           ---------------------

           alpha: float, double
                  default == 0.15
                  Teleportation parameter of the personalized PageRank linear system.
                  The smaller the more global the personalized PageRank vector is.

           rho:   float, double
                  defaul == 1.0e-5
                  Regularization parameter for the l1-norm of the model.

           For details of these parameters please refer to: 
           R. Andersen, F. Chung and K. Lang. Local Graph Partitioning using PageRank Vectors
           link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf

           max_iter: integer
                     default = 100000
                     Maximum number of iterations of ACL.
                     
           max_time: float, double
                     default = 100
                     Maximum time in seconds

           RETURNS
           -------

           The output can be accessed from the localCluster object that calls this function.

           node_embedding_acl: numpy array, float
                               Approximate personalized PageRank vector

           best_cluster_acl: list
                             A list of nodes that correspond to the cluster with the best 
                             conductance that was found by the algorithm.

           best_conductance_acl: float, double
                                 Conductance value that corresponds to the cluster with the best 
                                  conductance that was found by the algorithm.

           sweep_profile_acl: list of objects
                              A two dimensional list of objects. For example,
                              sweep_profile[0] contains a numpy array with all conductances for all
                              clusters that were calculated by sweep_cut.
                              sweep_profile[1] is a multidimensional list that contains the indices
                              of all clusters that were calculated by sweep_cut. For example,
                              sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                              that was calculated by sweep_cut. The set of indices in sweep_profile[1][5] also correspond 
                              to conductance in sweep_profile[0][5]. The number of clusters is unknwon apriori 
                              and depends on the data and that parameter setting of the algorithm.                     
        """ 
        self.node_embedding_acl = acl_list(ref_node, g, alpha = alpha, rho = rho, max_iter = max_iter, max_time = max_time)

        sweep = sweepCut()
        sweep.sweep_normalized(self.node_embedding_acl,g)

        self.best_cluster_acl = sweep.best_cluster
        self.best_conductance_acl = sweep.best_conductance
        self.sweep_profile_acl = sweep.sweep_profile
    
    def fista(self, ref_node, g, alpha = 0.15, rho = 1.0e-5, epsilon = 1.0e-6, max_iter = 10000, vol_G = -1, max_time = 100, cpp = True):
        """DESCRIPTION
           -----------

           Fast Iterative Soft Thresholding Algorithm (FISTA). This algorithm solves the l1-regularized
           personalized PageRank problem using an accelerated version of ISTA. It rounds the solution 
           using sweep cut.

           The l1-regularized personalized PageRank problem is defined as

           min rho*||p||_1 + <c,p> + <p,Q*p>

           where p is the PageRank vector, ||p||_1 is the l1-norm of p, rho is the regularization parameter 
           of the l1-norm, c is the right hand side of the personalized PageRank linear system and Q is the 
           symmetrized personalized PageRank matrix.    

           For details regarding ISTA please refer to: 
           K. Fountoulakis, F. Roosta-Khorasani, J. Shun, X. Cheng and M. Mahoney. Variational 
           Perspective on Local Graph Clustering. arXiv:1602.01886, 2017.
           arXiv link:https://arxiv.org/abs/1602.01886 

           PARAMETERS (mandatory)
           ----------------------

           ref_node: integer
                     The reference node, i.e., node of interest around which
                     we are looking for a target cluster.

           g: graph object

           PARAMETERS (optional)
           ---------------------

           alpha: float, double
                  default == 0.15
                  Teleportation parameter of the personalized PageRank linear system.
                  The smaller the more global the personalized PageRank vector is.

           rho:   float, double
                  defaul == 1.0e-5
                  Regularization parameter for the l1-norm of the model.

           For details of these parameters please refer to: K. Fountoulakis, F. Roosta-Khorasani, 
           J. Shun, X. Cheng and M. Mahoney. Variational Perspective on Local Graph Clustering. arXiv:1602.01886, 2017
           arXiv link:https://arxiv.org/abs/1602.01886 

           epsilon: float, double
                    default == 1.0e-6
                    Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.

           max_iter: integer
                     default = 10000
                     Maximum number of iterations of FISTA.
                     
           max_time: float, double
                     default = 100
                     Maximum time in seconds

           cpp: boolean
                default = True
                Use the faster C++ version of FISTA or not.

           RETURNS
           -------

           The output can be accessed from the localCluster object that calls this function.

           If cpp = False then the output is:

               node_embedding_fista: numpy array, float
                                     Approximate personalized PageRank vector

               best_cluster_fista: list
                                   A list of nodes that correspond to the cluster with the best 
                                   conductance that was found by FISTA.

               best_conductance:  float, double
                                  Conductance value that corresponds to the cluster with the best 
                                  conductance that was found by FISTA.

               sweep_profile_fista: list of objects
                                    A two dimensional list of objects. For example,
                                    sweep_profile[0] contains a numpy array with all conductances for all
                                    clusters that were calculated by sweep_cut.
                                    sweep_profile[1] is a multidimensional list that contains the indices
                                    of all clusters that were calculated by sweep_cut. For example,
                                    sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                                    that was calculated by sweep_cut. The set of indices in sweep_profile[1][5] also correspond 
                                    to conductance in sweep_profile[0][5]. The number of clusters is unknwon apriori 
                                    and depends on the data and that parameter setting of FISTA.

           If cpp = True then the output is:

               node_embedding_fista: numpy array, float
                                     Approximate personalized PageRank vector

               best_cluster_fista: list
                                   A list of nodes that correspond to the cluster with the best 
                                   conductance that was found by FISTA.

               best_conductance_fista: float, double
                                       Conductance value that corresponds to the cluster with the best 
                                       conductance that was found by FISTA.
        """
        sweep = sweepCut()

        if not cpp:
            self.node_embedding_fista = fista_dinput_dense(ref_node, g, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time)

            sweep.sweep_normalized(self.node_embedding_fista,g)

            self.best_cluster_fista = sweep.best_cluster
            self.best_conductance_fista = sweep.best_conductance
            self.sweep_profile_fista = sweep.sweep_profile
        else:

            uint_indptr = np.uint32(g.A.indptr) 
            uint_indices = np.uint32(g.A.indices)

            (not_converged,grad,self.node_embedding_fista) = proxl1PRaccel(uint_indptr, uint_indices, g.A.data, ref_node, g.d, g.d_sqrt, g.dn_sqrt, alpha = alpha, rho = rho, epsilon = epsilon, maxiter = max_iter, max_time = max_time)

            n = g.A.shape[0]

            sweep.sweep_cut_cpp(self.node_embedding_fista,g)  

            self.best_cluster_fista = sweep.best_cluster
            self.best_conductance_fista = sweep.best_conductance
        
    def ista(self, ref_node, g, alpha = 0.15, rho = 1.0e-5, epsilon = 1.0e-2, max_iter = 10000, max_time = 100):
        """DESCRIPTION
           -----------

           Iterative Soft Thresholding Algorithm (ISTA). This algorithm solves the l1-regularized
           personalized PageRank problem. It rounds the solution using sweep cut.

           The l1-regularized personalized PageRank problem is defined as

           min rho*||p||_1 + <c,p> + <p,Q*p>

           where p is the PageRank vector, ||p||_1 is the l1-norm of p, rho is the regularization parameter 
           of the l1-norm, c is the right hand side of the personalized PageRank linear system and Q is the 
           symmetrized personalized PageRank matrix.

           For details please refer to: 
           K. Fountoulakis, F. Roosta-Khorasani, J. Shun, X. Cheng and M. Mahoney. Variational 
           Perspective on Local Graph Clustering. arXiv:1602.01886, 2017.
           arXiv link:https://arxiv.org/abs/1602.01886 

           PARAMETERS (mandatory)
           ----------------------

           ref_node:  integer
                      The reference node, i.e., node of interest around which
                      we are looking for a target cluster.

           g:         graph object

           PARAMETERS (optional)
           ---------------------

           alpha: float, double
                  default == 0.15
                  Teleportation parameter of the personalized PageRank linear system.
                  The smaller the more global the personalized PageRank vector is.

           rho:   float, double
                  defaul == 1.0e-5
                  Regularization parameter for the l1-norm of the model.

           For details of these parameters please refer to: K. Fountoulakis, F. Roosta-Khorasani, 
           J. Shun, X. Cheng and M. Mahoney. Variational Perspective on Local Graph Clustering. arXiv:1602.01886, 2017
           arXiv link:https://arxiv.org/abs/1602.01886 

           epsilon: float, double
                    default == 1.0e-2
                    Tolerance for ISTA for solving the l1-regularized personalized PageRank problem.

           max_iter: integer
                     default = 10000
                     Maximum number of iterations of ISTA.
                     
           max_time: float, double
                     default = 100
                     Maximum time in seconds

           RETURNS
           ------- 
           
           The output can be accessed from the localCluster object that calls this function.

           node_embedding_ista: numpy array, float
                                Approximate personalized PageRank vector

           best_cluster_ista: list
                              A list of nodes that correspond to the cluster with the best 
                              conductance that was found by ISTA.

           best_conductance_ista: float, double
                                  Conductance value that corresponds to the cluster with the best 
                                  conductance that was found by ISTA.

           sweep_profile_ista: list of objects
                               A two dimensional list of objects. For example,
                               sweep_profile[0] contains a numpy array with all conductances for all
                               clusters that were calculated by sweep_cut.
                               sweep_profile[1] is a multidimensional list that contains the indices
                               of all clusters that were calculated by sweep_cut. For example,
                               sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                               that was calculated by sweep_cut. The number of clusters is unknwon apriori 
                               and depends on the data and that parameter setting of ISTA.                      
        """ 
        self.node_embedding_ista = ista_dinput_dense(ref_node, g, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time)

        sweep = sweepCut()
        sweep.sweep_normalized(self.node_embedding_ista,g)

        self.best_cluster_ista = sweep.best_cluster
        self.best_conductance_ista = sweep.best_conductance
        self.sweep_profile_fista = sweep.sweep_profile
    
    def page_rank_nibble(self, g, ref_node, vol, phi = 0.5, algorithm = 'fista', epsilon = 1.0e-2, max_iter = 10000, max_time = 100, cpp = True):
        """
           DESCRIPTION
           -----------

           Page Rank Nibble Algorithm. For details please refer to: 
           R. Andersen, F. Chung and K. Lang. Local Graph Partitioning using PageRank Vectors
           link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf
           The algorithm works on the connected component that the given reference node belongs to.

           PARAMETERS (mandatory)
           ----------------------

           g:         graph object       

           ref_node:  integer
                      The reference node, i.e., node of interest around which
                      we are looking for a target cluster.

           vol:       float, double
                      Lower bound for the volume of the output cluster.

           PARAMETERS (optional)
           ---------------------

           phi: float, double
                default == 0.5
                Target conductance for the output cluster.

           algorithm: string
                      default == 'fista'
                      Algorithm for spectral local graph clustering
                      Options: 'fista', 'ista', 'acl'.

           epsilon: float, double
                    default = 1.0e-2
                    Termination tolerance for l1-regularized PageRank, i.e., applies to FISTA and ISTA algorithms

           max_iter: integer
                     default = 10000
                     Maximum number of iterations of FISTA, ISTA or ACL.

           max_time: float, double
                     default = 100
                     Maximum time in seconds

           cpp: boolean
                default = True
                Use the faster C++ version of FISTA or not.

           RETURNS
           -------
           
           The output can be accessed from the localCluster object that calls this function.

           If cpp = False then the output is:

               node_embedding_nibble: numpy array, float 
                                      Approximate personalized PageRank vector

               best_cluster_nibble: list
                                    A list of nodes that correspond to the cluster with the best 
                                    conductance that was found by the algorithm.

               best_conductance_nibble: float
                                        Conductance value that corresponds to the cluster with the best 
                                        conductance that was found by the algorithm.

               sweep_profile_nibble: list of objects
                                     A two dimensional list of objects. For example,
                                     sweep_profile[0] contains a numpy array with all conductances for all
                                     clusters that were calculated by sweep_cut.
                                     sweep_profile[1] is a multidimensional list that contains the indices
                                     of all clusters that were calculated by sweep_cut. For example,
                                     sweep_profile[1][5] is a list that contains the indices of the 5th cluster
                                     that was calculated by sweep_cut. The set of indices in sweep_profile[1][5] also correspond 
                                     to conductance in sweep_profile[0][5]. The number of clusters is unknwon apriori 
                                     and depends on the data and that parameter setting of the algorithm.  

              volume_profile_nibble: list of objects
                                     A two dimensional list of objects which stores information about clusters
                                     which have volume larger than the input vol and les than 2/3 of the volume
                                     of the whole graph. For example, volume_profile[0] contains a list 
                                     with all conductances for all clusters that were calculated by sweep_cut and 
                                     also satisfy the previous volume constraint.
                                     volume_profile[1] is a multidimensional list that contains the indices
                                     of all clusters that were calculated by sweep_cut and also satisfy the previous 
                                     volume constraint. For example, volume_profile[1][5] is a list that contains the 
                                     indices of the 5th cluster that was calculated by sweep_cut and also satisfies 
                                     the previous volume constraint. The set of indices in volume_profile[1][5] also correspond 
                                     to conductance in volume_profile[0][5]. The number of clusters is unknwon apriori and 
                                     depends on the data and that parameter setting of the algorithm.

           If cpp = True then the output is:

               node_embedding_nibble: numpy array, float 
                                      Approximate personalized PageRank vector

               best_cluster_nibble: list
                                    A list of nodes that correspond to the cluster with the best 
                                    conductance that was found by the algorithm.

               best_conductance_nibble: float
                                        Conductance value that corresponds to the cluster with the best 
                                        conductance that was found by the algorithm.
        """ 
        n = g.A.shape[0]
        nodes = range(n)
        g_copy = g

        m = g_copy.A.count_nonzero()/2

        B = np.log2(m)

        if vol < 0:
            print("The input volume must be non-negative")
            return [], [], [], [], []
        if vol == 0:
            vol_user = 1
        else:
            vol_user = vol

        b = 1 + np.log2(vol_user)

        b = min(b,B)

        alpha = (phi**2)/(225*np.log(100*np.sqrt(m)))

        rho = (1/(2**b))*(1/(48*B))

        if algorithm == 'fista':
            if not cpp:
                p = fista_dinput_dense(ref_node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time)
            else:
                uint_indptr = np.uint32(g.A.indptr) 
                uint_indices = np.uint32(g.A.indices)

                (not_converged,grad,p) = proxl1PRaccel(uint_indptr, uint_indices, g.A.data, ref_node, g.d, g.d_sqrt, g.dn_sqrt, alpha = alpha, rho = rho, epsilon = epsilon, maxiter = max_iter, max_time = max_time)
        elif algorithm == 'ista':
            p = ista_dinput_dense(ref_node, g_copy, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time)
        elif algorithm == 'acl':
            p = acl_list(ref_node, g_copy, alpha = alpha, rho = rho, max_iter = max_iter, max_time = max_time)
        else:
            print("There is no such algorithm provided")
            return [], [], [], []

        sweep = sweepCut()    

        if not cpp:
            sweep.sweep_normalized(p,g_copy,vol)

            for i in range(len(sweep.sweep_profile[0])):
                sweep.sweep_profile[1][i] = [nodes[j] for j in sweep.sweep_profile[1][i]]

            for i in range(len(sweep.volume_profile[0])):
                sweep.volume_profile[1][i] = [nodes[j] for j in sweep.volume_profile[1][i]]

            sweep.best_cluster = [nodes[i] for i in sweep.best_cluster]

            self.node_embedding_nibble = p
            self.best_cluster_nibble = sweep.best_cluster
            self.best_conductance_nibble = sweep.best_conductance
            self.sweep_profile_nibble = sweep.sweep_profile
            self.volume_profile_nibble = sweep.volume_profile
        else:
            n = g.A.shape[0]

            sweep.sweep_cut_cpp(p,g)     

            self.node_embedding_nibble = p
            self.best_cluster_nibble = sweep.best_cluster
            self.best_conductance_nibble = sweep.best_conductance
