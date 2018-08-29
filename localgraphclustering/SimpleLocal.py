from typing import *
import numpy as np
from .cpp import *
from .GraphLocal import GraphLocal

def SimpleLocal(G, ref_nodes,
                delta: float = 0.3):
    """
    A Simple and Strongly-Local Flow-Based Method for Cut Improvement.
    For details please refer to: Veldt, Gleich and Mahoney (2016).
    link: https://arxiv.org/abs/1605.08490

    Parameters
    ----------

    inputs: Sequence[Graph]

    ref_node: Sequence[Sequence[int]]
        The reference node, i.e., node of interest around which
        we are looking for a target cluster.

    delta: float
        locality parameter
            
    Returns
    -------
        
    It returns in a list of length 2 with the following:
        
    output 0: list
        Stores indices of the best clusters found by the last called rounding procedure.
           
    output 1: float
        Stores the value of the best conductance found by the last called rounding procedure.

    """ 
    n = G.adjacency_matrix.shape[0]
    (actual_length,actual_xids) = SimpleLocal_cpp(n,G.ai,G.aj,len(ref_nodes),ref_nodes,delta)

    return [actual_xids, G.compute_conductance(actual_xids)]
