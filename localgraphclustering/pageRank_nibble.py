from typing import *
import numpy as np
from .approximate_PageRank import approximate_PageRank

def PageRank_nibble(g,ref_nodes,
                    vol: float = 100,
                    phi: float = 0.5,
                    method: str = 'acl',
                    epsilon: float = 1.0e-2,
                    iterations: int = 10000,
                    timeout: int = 100,
                    ys: Sequence[float] = None, 
                    cpp: bool = True):
    """
    Page Rank Nibble Algorithm. For details please refer to: 
    R. Andersen, F. Chung and K. Lang. Local Graph Partitioning using PageRank Vectors
    link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf
    The algorithm works on the connected component that the given reference node belongs to.

    This method stores the results in the class attribute page_rank_nibble_transformation.

    Parameters (mandatory)
    ----------------------

    g: graph object       

    ref_nodes: Sequence[int]
        A sequence of reference nodes, i.e., nodes of interest around which
        we are looking for a target cluster.

    Parameters (optional)
    ---------------------

    vol: float, double
        Lower bound for the volume of the output cluster.

    phi: float64
        Default == 0.5
        Target conductance for the output cluster.

    method: string
        Default = 'acl'
        Which method to use.
        Options: 'l1reg', 'acl'.

    iterations: int
        default = 10000
        Maximum number of iterations.

    timeout: float64
        default = 100
        Maximum time in seconds

    cpp: bool
        default = True
        Use the faster C++ version or not.

    Extra parameters for "l1reg" (optional)
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
    n = g.adjacency_matrix.shape[0]
    nodes = range(n)

    m = g.adjacency_matrix.count_nonzero()/2

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

    p = approximate_PageRank(g, ref_nodes, timeout = timeout, iterations = iterations, alpha = alpha,
                             rho = rho, epsilon = epsilon, ys = ys, cpp = cpp, method = method)


    return p