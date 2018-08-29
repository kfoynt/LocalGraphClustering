from typing import *
import numpy as np
from .cpp import *

def approximate_PageRank_weighted(G,
                                  ref_nodes,
                                  iterations: int = 100000,
                                  alpha: float = 0.15,
                                  rho: float = 1.0e-6):
    """
    Computes an approximate PageRank vector on a weighted graph. Uses the modified Andersen Chung and Lang (ACL) Algorithm. 
    Now, the diffusion from one node to its neighbors is proportional to the edge weight.
    For details please refer to: R. Andersen, F. Chung and K. Lang. Local Graph Partitioning 
    using PageRank Vectors
    link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf. 

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
        Default = 1000
        Maximum number of iterations of ACL algorithm.
            
    Returns
    -------
        
    An np.ndarray (2D embedding) of the nodes for each graph.

    """ 
            
    #print("Uses the weighted Andersen Chung and Lang (ACL) Algorithm.")
    n = G.adjacency_matrix.shape[0]
    (length,xids,values) = aclpagerank_weighted_cpp(n,G.ai,G.aj,G.adjacency_matrix.data,alpha,rho,
                ref_nodes,1,iterations)
    p = np.zeros(n)
    p[xids] = values

    return p
