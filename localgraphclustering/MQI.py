from typing import *
import numpy as np
from .cpp import *

def MQI(G, ref_nodes):
    """
    Max Flow Quotient Cut Improvement (MQI) for improving either the expansion or the 
    conductance of cuts of weighted or unweighted graphs.
    For details please refer to: Lang and Rao (2004). Max Flow Quotient Cut Improvement (MQI)
    link: https://link.springer.com/chapter/10.1007/978-3-540-25960-2_25

    Parameters
    ----------------------

    G: GraphLocal      

    ref_nodes: Sequence[int]
        A sequence of reference nodes, i.e., nodes of interest around which
        we are looking for a target cluster.
            
    Returns
    -------
        
    It returns in a list of length 2 with the following:
        
    output 0: list
        Stores indices of the best clusters found by the last called rounding procedure.
           
    output 1: float
        Stores the value of the best conductance found by the last called rounding procedure.

    """ 
    n = G.adjacency_matrix.shape[0]

    R = list(set(ref_nodes))

    nR = len(R)

    (actual_length,actual_xids) = MQI_cpp(n,G.ai,G.aj,nR,R)

    return [actual_xids, G.compute_conductance(actual_xids)]
