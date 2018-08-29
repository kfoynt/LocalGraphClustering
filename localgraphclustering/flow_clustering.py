from typing import *
import numpy as np
from .MQI import MQI
from .SimpleLocal import SimpleLocal
from .capacity_releasing_diffusion import capacity_releasing_diffusion
from .GraphLocal import GraphLocal
import warnings

def flow_clustering(G, ref_nodes,
                    U: int = 3,
                    h: int = 10,
                    w: int = 2,
                    iterations: int = 20,
                    delta: float = 0.3,
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
        Options: "mqi", "sl", "crd"

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
            warnings.warn("The weights of the graph will be discarded. Use \"crd\" if you want to keep them.")
        return MQI(G,ref_nodes)
    elif method == "crd":
        return capacity_releasing_diffusion(G,ref_nodes,U=U,h=h,w=w,iterations=iterations)
    elif method == "sl":
        if G._weighted:
            warnings.warn("The weights of the graph will be discarded. Use \"crd\" if you want to keep them.")
        return SimpleLocal(G,ref_nodes,delta=delta)
    else:
        raise Exception("Unknown method, available methods are \"mqi\", \"crd\", \"sl\".")


