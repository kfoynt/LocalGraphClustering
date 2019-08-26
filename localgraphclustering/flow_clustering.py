from typing import *
import numpy as np
from .MQI import MQI
from .MQI_weighted import MQI_weighted
from .SimpleLocal import SimpleLocal
from .SimpleLocal_weighted import SimpleLocal_weighted
from .capacity_releasing_diffusion import capacity_releasing_diffusion
from .GraphLocal import GraphLocal
import warnings

def flow_clustering(G, ref_nodes,
                    U: int = 3,
                    h: int = 10,
                    w: int = 2,
                    iterations: int = 20,
                    delta: float = 0.3,
                    relcondflag: bool = True,
                    check_connectivity: bool = True,
                    method: str = "mqi"):
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
        Options: "mqi", "mqi_weighted", "sl", "sl_weighted","crd"

    Optional parameters for "sl" and "sl_weighted"
    ----------------------

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
    
    if method == "mqi":
        if G._weighted:
            warnings.warn("The weights of the graph will be discarded. Use \"crd\" or \"mqi_weighted\" if you want to keep them.")
        return MQI(G,ref_nodes)
    if method == "mqi_weighted":
        return MQI_weighted(G,ref_nodes)
    elif method == "crd":
        return capacity_releasing_diffusion(G,ref_nodes,U=U,h=h,w=w,iterations=iterations)
    elif method == "sl":
        if G._weighted:
            warnings.warn("The weights of the graph will be discarded. Use \"crd\" if you want to keep them.")
        return SimpleLocal(G,ref_nodes,delta=delta,relcondflag=relcondflag,check_connectivity=check_connectivity)
    elif method == "sl_weighted":
        return SimpleLocal_weighted(G,ref_nodes,delta=delta,relcondflag=relcondflag,check_connectivity=check_connectivity)
    else:
        raise Exception("Unknown method, available methods are \"mqi\", \"mqi_weighted\", \"crd\", \"sl\", \"sl_weighted\".")


