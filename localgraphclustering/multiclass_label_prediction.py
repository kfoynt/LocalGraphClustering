from typing import *
import numpy as np
from scipy import sparse as sp
from .algorithms import fista_dinput_dense
from .cpp import *
from .GraphLocal import GraphLocal

def multiclass_label_prediction(g: GraphLocal,
                                timeout: float = 100, 
                                iterations: int = 1000,
                                labels: np.ndarray = [],
                                alpha: float = 0.15,
                                rho: float = 1.0e-6,
                                epsilon: float = 1.0e-2,
                                cpp: bool = True):
    """
    This function predicts labels for unlabelled nodes. For details refer to:
    D. Gleich and M. Mahoney. Variational 
    Using Local Spectral Methods to Robustify Graph-Based Learning Algorithms. SIGKDD 2015.
    https://www.stat.berkeley.edu/~mmahoney/pubs/robustifying-kdd15.pdf

    Parameters (mandatory)
    ----------------------
        
    inputs: Sequence[Graph]

    labels: list of lists
        Each list of this list corresponds to indices of nodes that are assumed to belong in
        a certain class. For example, list[i] is a list of indices of nodes that are assumed to 
        belong in class i.

    Parameters (optional)
    ---------------------

    alpha: float, double
        Default == 0.15
        Teleportation parameter of the personalized PageRank linear system.
        The smaller the more global the personalized PageRank vector is.

    rho: float, double
        Defaul == 1.0e-10
        Regularization parameter for the l1-norm of the model.

    epsilon: float, double
        Default == 1.0e-2
        Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.

    iterations: int
        Default = 100000
        Maximum number of iterations of FISTA algorithm.
                     
    timeout: float
        Default = 100
        Maximum time in seconds.

    cpp: bool
        default = True
        Use the faster C++ version of FISTA or not.

    Returns
    -------

    output 0: list of indices that holds the class for each node.
        For example classes[i] is the class of node i.

    output 1: list of lists. Each componenent of the list is a list that holds the rank
        of the nodes for each class. For details see [1].

    output 2: a list of numpy arrays. Each array in this list corresponds to the diffusion vector
        returned by personalized PageRank for each rank. For details see [1].

    [1] D. Gleich and M. Mahoney. Variational 
    Using Local Spectral Methods to Robustify Graph-Based Learning Algorithms. SIGKDD 2015.
    https://www.stat.berkeley.edu/~mmahoney/pubs/robustifying-kdd15.pdf
    """  
    n = g.adjacency_matrix.shape[0]
    
    output = [[],[],[]]

    for labels_i in labels: 
        
        if not cpp:
            output_fista = fista_dinput_dense(labels_i, g, alpha = alpha, rho = rho, epsilon = epsilon, iterations = iterations, timeout = timeout)
        else: 
            (not_converged,grad,output_fista) = proxl1PRaccel_cpp(g.ai, g.aj, g.adjacency_matrix.data, labels_i, g.d, g.d_sqrt, g.dn_sqrt, alpha = alpha, rho = rho, epsilon = epsilon, maxiter = iterations, max_time = timeout)
        
        p = np.zeros(n)
        for i in range(n):
            p[i] = output_fista[i]
        
        output[0].append(p)
        
        index = (-p).argsort(axis=0)
        rank = np.empty(n, int)
        rank[index] = np.arange(n)
        
        output[1].append(rank)
        
    l_labels = len(labels)
    
    for i in range(n):
        min_rank = n+1
        class_ = l_labels + 1
        for j in range(l_labels):
            rank = output[1][j][i]
            if rank < min_rank:
                min_rank = rank
                class_ = j
        output[2].append(class_)
        
    return output