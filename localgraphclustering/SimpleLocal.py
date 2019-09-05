from typing import *
import numpy as np
from .cpp import *
from .GraphLocal import GraphLocal

def SimpleLocal(G, ref_nodes,
                delta: float = 0.3, relcondflag: bool = True,
                check_connectivity: bool = True):
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

    relcondflag: bool, default is True
        a boolean flag indicating whether to compute the relative
        conductance score or the exact conductance score for each
        intermediate improved set. Choosing false (i.e. updating with
        exact conductance) will sometimes lead to fewer iterations and
        lower conductance output, but will not actually minimize the
        relative conductance or seed penalized conductance. Choosing true
        will guarantee the returned set is connected.
        
    check_connectivity: bool, default is True
        a boolean flag indicating whether to do ax extra DFS to ensure the
        returned set is connected. Only effective when "relcondflag" is set
        to be True.

    Returns
    -------
        
    It returns in a list of length 2 with the following:
        
    output 0: list
        Stores indices of the best clusters found by the last called rounding procedure.
           
    output 1: float
        Stores the value of the best conductance found by the last called rounding procedure.

    """ 
    n = G.adjacency_matrix.shape[0]
    (actual_length,actual_xids) = SimpleLocal_cpp(n,G.ai,G.aj,len(ref_nodes),ref_nodes,delta,relcondflag)
    # Use DFS to guarantee the returned set is connected
    if relcondflag and check_connectivity:
        stk = [actual_xids[0]]
        curr_set = set(actual_xids)
        ret_set = [actual_xids[0]]
        visited = set([actual_xids[0]])
        while len(stk) > 0:
            curr_node = stk.pop()
            for i in range(G.ai[curr_node],G.ai[curr_node+1]):
                if G.aj[i] in curr_set and G.aj[i] not in visited:
                    stk.append(G.aj[i])
                    ret_set.append(G.aj[i])
                    visited.add(G.aj[i])
        actual_xids = np.array(ret_set)
    if len(actual_xids) > G._num_vertices/2:
        actual_xids = np.array(list(set(range(G._num_vertices)).difference(actual_xids)))
        
    return [actual_xids, G.compute_conductance(actual_xids)]
