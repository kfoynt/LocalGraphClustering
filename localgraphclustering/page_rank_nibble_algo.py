import numpy as np
from localgraphclustering.proxl1PRaccel import proxl1PRaccel
from localgraphclustering.fista_dinput_dense import fista_dinput_dense
from localgraphclustering.acl_list import acl_list

def page_rank_nibble_algo(g,ref_node,vol,phi = 0.5,algorithm='fista',epsilon=1.0e-2,max_iter=10000,max_time=100,cpp=True):
    """
    Page Rank Nibble Algorithm. For details please refer to: 
    R. Andersen, F. Chung and K. Lang. Local Graph Partitioning using PageRank Vectors
    link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf
    The algorithm works on the connected component that the given reference node belongs to.

    This method stores the results in the class attribute page_rank_nibble_transformation.

    Parameters (mandatory)
    ----------------------

    g: graph object       

    ref_node:  integer
        The reference node, i.e., node of interest around which
        we are looking for a target cluster.

    vol: float, double
        Lower bound for the volume of the output cluster.

    Parameters (optional)
    ---------------------

    phi: float64
        Default == 0.5
        Target conductance for the output cluster.

    algorithm: string
        Default == 'fista'
        Algorithm for spectral local graph clustering
        Options: 'fista', 'ista', 'acl'.

    epsilon: float64
        Default = 1.0e-2
        Termination tolerance for l1-regularized PageRank, i.e., applies to FISTA and ISTA algorithms

    max_iter: int
        default = 10000
        Maximum number of iterations of FISTA, ISTA or ACL.

    max_time: float64
        default = 100
        Maximum time in seconds

    cpp: bool
        default = True
        Use the faster C++ version of FISTA or not.
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

    if algorithm == 'fista':
        if not cpp:
            p = fista_dinput_dense(ref_node, g, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time)
        else:
            uint_indptr = np.uint32(g.adjacency_matrix.indptr) 
            uint_indices = np.uint32(g.adjacency_matrix.indices)

            (not_converged,grad,p) = proxl1PRaccel(uint_indptr, uint_indices, g.adjacency_matrix.data, ref_node, g.d, g.d_sqrt, g.dn_sqrt, g.lib, alpha = alpha, rho = rho, epsilon = epsilon, maxiter = max_iter, max_time = max_time)
            p = np.abs(p)
    elif algorithm == 'ista':
        p = ista_dinput_dense(ref_node, g, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time)
    elif algorithm == 'acl':
        p = acl_list(ref_node, g, alpha = alpha, rho = rho, max_iter = max_iter, max_time = max_time)
    else:
        raise Exception("There is no such algorithm provided")

    return p