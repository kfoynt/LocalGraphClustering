from typing import *
import numpy as np
from .fiedler import fiedler
from .sweep_cut import sweep_cut
from .approximate_PageRank import approximate_PageRank
from .GraphLocal import GraphLocal
import warnings

def spectral_clustering(G, ref_nodes,
                        timeout: float = 100,
                        iterations: int = 100000,
                        alpha: float = 0.15,
                        rho: float = 1.0e-6,
                        epsilon: float = 1.0e-2,
                        ys: Sequence[float] = None,
                        vol: float = 100,
                        phi: float = 0.5,
                        method: str = "acl"):
    """
    Provide a simple interface to do spectral based clustering.

    Parameters
    ----------------------

    G: GraphLocal

    ref_nodes: Sequence[int]
        A sequence of reference nodes, i.e., nodes of interest around which
        we are looking for a target cluster.

    method: str
        Which method to use for the nodes embedding.
        Options: "acl", "l1reg", "nibble", "fiedler", "fiedler_local"

    Extra parameters for "acl" and "l1reg" (optional)
    -------------------------------------------------

    alpha: float
        Default == 0.15
        Teleportation parameter of the personalized PageRank linear system.
        The smaller the more global the personalized PageRank vector is.

    rho: float
        Defaul == 1.0e-6
        Regularization parameter for the l1-norm of the model.

    iterations: int
        Default = 100000
        Maximum number of iterations of ACL algorithm.

    timeout: float
        Default = 100
        Maximum time in seconds

    Extra parameters for "l1reg" (optional)
    ----------------------------------------

    epsilon: float
        Default == 1.0e-2
        Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.

    ys: Sequence[float]
        Defaul == None
        Initial solutions for l1-regularized PageRank algorithm.
        If not provided then it is initialized to zero.
        This is only used for the C++ version of FISTA.

    Extra parameters for "nibble" (optional)
    ----------------------------------------

    vol: float
        Lower bound for the volume of the output cluster.

    phi: float
        Default == 0.5
        Target conductance for the output cluster.

    Returns
    -------

    It returns in a list of length 2 with the following:

    output 0: list
        Stores indices of the best clusters found by the last called rounding procedure.

    output 1: float
        Stores the value of the best conductance found by the last called rounding procedure.
    """

    if G._weighted:
        warnings.warn("The weights of the graph will be discarded. Use approximate_PageRank_weighted instead if you want to keep the edge weights.")

    if method == "acl" or method == "acl_weighted" or method == "l1reg":
        p = approximate_PageRank(G,ref_nodes,timeout = timeout, iterations = iterations, alpha = alpha,
            rho = rho, epsilon = epsilon, method = method, ys = ys)
    elif method == "nibble":
        p = PageRank_nibble(G,ref_nodes,vol = vol,phi = phi,epsilon = epsilon,iterations = iterations,timeout = timeout)
    elif method == "fiedler":
        warnings.warn("ref_nodes will be disgarded since we are computing a global fiedler vector.")
        p = fiedler(G)
    elif method == "fiedler_local":
        p = fiedler(G,ref_nodes)
    else:
        raise Exception("Unknown method, available methods are \"acl\", \"acl_weighted\", \"l1reg\", \"nibble\", \"fiedler\", \"fiedler_local\".")

    output = sweep_cut(G,p)

    return output
