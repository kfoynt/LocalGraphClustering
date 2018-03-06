from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
import pandas as pd

from localgraphclustering.NCPData import NCPData
from localgraphclustering.NCPAlgo import *

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=pd.DataFrame)


class Ncp(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the Ncp class.
        """

        super().__init__()

    def produce(self, 
                input: Input, 
                method: str,
                ratio: float = 0.3,
                timeout: float = 1000,
                nthreads: int = 4,
                do_largest_component: bool = True,
                U: int = 3,
                h: int = 10,
                w: int = 2,
                epsilon: float = 1.0e-1,
                iterations: int = 20,
                ) -> Output:
        """
        Network Community Profile for the largest connected components of the graph. For details please refer to: 
        Jure Leskovec, Kevin J Lang, Anirban Dasgupta, Michael W Mahoney. Community structure in 
        large networks: Natural cluster sizes and the absence of large well-defined clusters.
        The NCP is computed for the largest connected component of the given graph.

        Parameters
        ----------  
        
        input: Graph
            Given graph whose network community Profile needs to be computed.

        method: str
            Choose either Capacity Releasing Diffusion or Max Flow Quotient Cut Improvement as the local clustering
            method, must be "crd", "mqi", "l1reg" or "approxPageRank".

        ratio: float
            Ratio of nodes to be used for computation of NCP.
            It should be between 0 and 1.

        Parameters (optional)
        ---------------------

        timeout: float
            default = 1000
            Maximum time in seconds for NCP calculation.

        nthreads: int
            default = 4
            Choose the number of threads used for NCP calculation
            
        do_largest_component: bool
            default = True
            If true it computes the NCP for the largest connected component.
            This task might double the required memory. If False then it computes
            the NCP for the given graph. This taks has minimal memory requirements.

        U: integer
            default == 3
            The net mass any edge can be at most.
          
        h: integer
            defaul == 10
            The label of any node can have at most.
          
        w: integer
            default == 2
            Multiplicative factor for increasing the capacity of the nodes at each iteration.

        epsilon: float
            default == 0.1
            Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.
          
        iterations: integer
            default = 20, 1000, 10000
            Maximum number of iterations of Capacity Releasing Diffusion Algorithm, l1_regularized Pagerank and Approximate PageRank.

        Returns
        -------
        
        df: pandas.DataFrame
            The output can be used as the parameter of NCPPlots to plot all sorts of graph.
        """ 
        G = input
        G.compute_statistics()
        ncp = NCPData(G,do_largest_component=do_largest_component)      
        if method == "crd":
            ncp.default_method = lambda G,R: crd_wrapper(G,R,w=w, U=U, h=h, iterations=iterations)
            ncp.add_random_neighborhood_samples(ratio=ratio,nthreads=nthreads,timeout=timeout)
            ncp.add_random_node_samples(ratio=ratio,nthreads=nthreads,timeout=timeout)
        elif method == "mqi":
            ncp.default_method = lambda G,R: mqi_wrapper(G,R)
            ncp.add_random_neighborhood_samples(ratio=ratio,nthreads=nthreads,timeout=timeout)
        elif method == "l1reg":
            ncp.default_method = lambda G,R: l1reg_wrapper(G,R,epsilon=epsilon)
            ncp.add_random_node_samples(ratio=ratio,nthreads=nthreads,timeout=timeout)
        elif method == "approxPageRank":  
            ncp.default_method = lambda G,R: approxPageRank_wrapper(G,R)
            ncp.add_random_node_samples(ratio=ratio,nthreads=nthreads,timeout=timeout)
        else:
            raise(ValueError("Must specify a method (crd, mqi or l1reg)."))
        df = ncp.as_data_frame()

        return df
        

