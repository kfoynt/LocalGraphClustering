from typing import *
import numpy as np
from .algorithms import acl_list
from .algorithms import fista_dinput_dense
from .cpp import *
import warnings

import time

def approximate_PageRank(G,
                         ref_nodes,
                         timeout: float = 100,
                         iterations: int = 1000000,
                         alpha: float = 0.15,
                         rho: float = 1.0e-6,
                         epsilon: float = 1.0e-2,
                         ys: Sequence[float] = None,
                         cpp: bool = True,
                         normalize: bool = True,
                         normalized_objective: bool = True,
                         method: str = "acl",
                         use_distribution: bool = False,
                         distribution: list = []):
    """
    Computes PageRank vector locally.
    --------------------------------

    When method is "acl" or "acl_weighted":

    Uses the Andersen Chung and Lang (ACL) Algorithm. For details please refer to:
    R. Andersen, F. Chung and K. Lang. Local Graph Partitioning using PageRank Vectors
    link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf.

    When method is "l1reg":

    Uses the Fast Iterative Soft Thresholding Algorithm (FISTA). This algorithm solves the l1-regularized
    personalized PageRank problem.

    The l1-regularized personalized PageRank problem is defined as

    min rho*||p||_1 + <c,p> + <p,Q*p>

    where p is the PageRank vector, ||p||_1 is the l1-norm of p, rho is the regularization parameter
    of the l1-norm, c is the right hand side of the personalized PageRank linear system and Q is the
    symmetrized personalized PageRank matrix.

    For details please refer to:
    K. Fountoulakis, F. Roosta-Khorasani, J. Shun, X. Cheng and M. Mahoney. Variational
    Perspective on Local Graph Clustering. arXiv:1602.01886, 2017.
    arXiv link:https://arxiv.org/abs/1602.01886

    When method is "l1reg-rand":

    Uses a randomized proximal coordinate descent method.

    Parameters
    ----------

    G: GraphLocal

    ref_nodes: Sequence[int]
        A sequence of reference nodes, i.e., nodes of interest around which
        we are looking for a target cluster.

    Parameters (optional)
    ---------------------

    alpha: float
        Default == 0.15
        Teleportation parameter of the personalized PageRank linear system.
        The smaller the more global the personalized PageRank vector is.

    rho: float
        Defaul == 1.0e-6
        Regularization parameter for the l1-norm of the model.

    iterations: int
        Default = 1000000
        Maximum number of iterations of ACL algorithm.

    timeout: float
        Default = 100
        Maximum time in seconds

    method: string
        Default = 'acl'
        Which method to use.
        Options: 'l1reg', 'acl'.

    normalize: bool
        Default = True
        Normalize the output to be directly input into sweepcut routines.

    normalized_objective: bool
        Default = True
        Use normalized Laplacian in the objective function, works only for "method=l1reg" and "cpp=True"

    cpp: bool
        default = True
        Use the faster C++ version or not.

    Extra parameters for "l1reg" or "l1reg-rand" (optional)
    -------------------------------------------

    epsilon: float64
        Default == 1.0e-2
        Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.

    ys: Sequence[float]
        Defaul == None
        Initial solutions for l1-regularized PageRank algorithm.
        If not provided then it is initialized to zero.
        This is only used for the C++ version of FISTA.


    Returns
    -------

    An np.ndarray (1D embedding) of the nodes.
    """

    if G._weighted and method not in ["acl_weighted","l1reg-rand","l1reg"]:
        warnings.warn("The weights of the graph will be discarded. Use approximate_PageRank_weighted or l1reg-rand instead if you want to keep the edge weights.")
    if method == "acl":
        #print("Uses the Andersen Chung and Lang (ACL) Algorithm.")
        if ys != None:
            warnings.warn("\"acl\" doesn't support initial solutions, please use \"l1reg\" instead.")
        if cpp:
            
#             start = time.time()
            
            n = G.adjacency_matrix.shape[0]
            (length,xids,values) = aclpagerank_cpp(n,G.ai,G.aj,alpha,rho,ref_nodes,iterations)
            # TODO, implement this in the C++ function
            if normalize:
                for i in range(len(xids)): # we can't use degrees because it could be weighted
                    values[i] /= (G.ai[xids[i]+1]-G.ai[xids[i]])
                    
#             end = time.time()
#             print(" Elapsed time acl with rounding: ", end - start)
            
            return (xids,values)
        else:
            return acl_list(ref_nodes, G, alpha = alpha, rho = rho, max_iter = iterations, max_time = timeout)
    elif method == "acl_weighted":
        if ys != None:
            warnings.warn("\"acl_weighted\" doesn't support initial solutions, please use \"l1reg\" instead.")
        if cpp:
            n = G.adjacency_matrix.shape[0]
            (length,xids,values) = aclpagerank_weighted_cpp(n,G.ai,G.aj,G.adjacency_matrix.data,alpha,rho,
                ref_nodes,iterations)
            if normalize:
                values *= G.dn[xids] # those are inverse degrees
            return (xids,values)
        else:
            raise Exception("There is only C++ version for acl weighted.")
    elif method == "l1reg" or method == "l1reg-rand":
        #print("Uses the Fast Iterative Soft Thresholding Algorithm (FISTA).")
        # TODO fix the following warning
        # warnings.warn("The normalization of this routine hasn't been adjusted to the new system yet")
        
#         start = time.time()
        
        algo = proxl1PRaccel_cpp if method == "l1reg" else proxl1PRrand_cpp
        if cpp:
            if method == "l1reg":
                p = algo(G.ai, G.aj, G.adjacency_matrix.data, ref_nodes, G.d, G.d_sqrt, G.dn_sqrt, alpha = alpha,
                                     rho = rho, epsilon = epsilon, maxiter = iterations, max_time = timeout, normalized_objective = normalized_objective, 
                                     use_distribution = use_distribution, distribution = distribution)[2]
            else:
                (length,xids,values) = algo(G.ai, G.aj, G.adjacency_matrix.data, ref_nodes, G.d, G.d_sqrt, G.dn_sqrt, alpha = alpha,
                                     rho = rho, epsilon = epsilon, maxiter = iterations, max_time = timeout, normalized_objective = normalized_objective)
        else:
            p = fista_dinput_dense(ref_nodes, G, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = iterations, max_time = timeout)
        # convert result to a sparse vector
#         nonzeros = np.count_nonzero(p)
    
    
#         end = time.time()
#         print(" Elapsed time l1reg with rounding: ", end - start)
            
#         start = time.time()
        
#         idx = np.nonzero(p)[0]
        
#         end = time.time()
#         print(" Elapsed time one: ", end - start) 
        
#         start = time.time()
#         vals = p[idx]
        
#         end = time.time()
#         print(" Elapsed time two: ", end - start) 
        
#         start = time.time()
        if normalize:
            if method == "l1reg":
                # convert result to a sparse vector
                nonzeros = np.count_nonzero(p)
                xids = np.zeros(nonzeros,dtype=np.dtype(G.aj[0]))
                values = np.zeros(nonzeros,dtype=np.float64)
                it = 0
                for i in range(len(p)):
                    if p[i] != 0:
                        xids[it] = i
                        values[it] = p[i]*1.0 * G.dn[i] if normalize else p[i]
                        it += 1
            else:
                values = np.multiply(G.dn[xids], values)
                
        return (xids,values)
    else:
        raise Exception("Unknown method, available methods are \"acl\" or \"acl_weighted\" or \"l1reg\" or \"l1reg-rand\".")
