from typing import *
import numpy as np
from .cpp import *
from .GraphLocal import GraphLocal

def capacity_releasing_diffusion(G,ref_nodes,
                                 U: int = 3,
                                 h: int = 10,
                                 w: int = 2,
                                 iterations: int = 20):

    """
    Description
    -----------
       
    Algorithm Capacity Releasing Diffusion for local graph clustering. This algorithm uses 
    a flow based method to push excess flow out of nodes. The algorithm is in worst-case 
    faster and stays more local than classical spectral diffusion processes.
    For more details please refere to: D. Wang, K. Fountoulakis, M. Henzinger, M. Mahoney 
    and S. Rao. Capacity Releasing Diffusion for Speed and Locality. ICML 2017.
    arXiv link: https://arxiv.org/abs/1706.05826
       
    Parameters (mandatory)
    ----------------------

    G: GraphLocal      

    ref_nodes: Sequence[int]
        A sequence of reference nodes, i.e., nodes of interest around which
        we are looking for a target cluster.
       
    Parameters (optional)
    --------------------
      
    U: integer
        default == 3
        The maximum flow that can be send out of a node for the push/relabel algorithm.
          
    h: integer
        defaul == 10
        The maximum flow that an edge can handle.
          
    w: integer
        default == 2
        Multiplicative factor for increasing the capacity of the nodes at each iteration.
          
    iterations: integer
        default = 20
        Maximum number of iterations of Capacity Releasing Diffusion Algorithm.

       
    Returns
    -------
       
    It returns in a list of length 2 with the following:
        
    output 0: list
        Stores indices of the best clusters found by the last called rounding procedure.
           
    output 1: float
        Stores the value of the best conductance found by the last called rounding procedure.
       
    Printing statements (warnings)
    ------------------------------
       
    Too much excess: Means that push/relabel cannot push the excess flow out of the nodes.
                     This might indicate that a cluster has been found. In this case the best
                     cluster in terms of conductance is returned.
                       
    Too much flow: Means that the algorithm has touched about a third of the whole given graph.
                   The algorithm is terminated in this case and the best cluster in terms of 
                   conductance is returned. 
    """

    n = G.adjacency_matrix.shape[0]
    actual_xids = capacity_releasing_diffusion_cpp(n,G.ai,G.aj,np.float64(G.adjacency_matrix.data),
                U,h,w,iterations,ref_nodes)

    return [actual_xids, G.compute_conductance(actual_xids)]