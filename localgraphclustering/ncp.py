from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph

from localgraphclustering.ncp_algo import ncp_algo

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class Ncp(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the Ncp class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input], 
                ratio: float = 0.3,
                timeout: float = 100, 
                timeout_ncp = 1000,
                iterations: int = 1000,
                epsilon: float = 1.0e-1,
                nthreads: int = 20,
                multi_threads: bool = True
                ) -> Sequence[Output]:
        """
        Network Community Profile for all connected components of the graph. For details please refer to: 
        Jure Leskovec, Kevin J Lang, Anirban Dasgupta, Michael W Mahoney. Community structure in 
        large networks: Natural cluster sizes and the absence of large well-defined clusters.
        The NCP is computed for each connected component of the given graph(s).

        Parameters
        ----------  

        ratio: float
            Ratio of nodes to be used for computation of NCP.
            It should be between 0 and 1.

        Parameters (optional)
        ---------------------

        epsilon: float
            default = 1.0e-1
            Termination tolerance for l1-regularized PageRank solver.

        iterations: int
            default = 10000
            Maximum number of iterations of l1-regularized PageRank solver.

        timeout_ncp: float
            default = 1000
            Maximum time in seconds for NCP calculation.

        timeout: float
            default = 10
            Maximum time in seconds for each algorithm run during the NCP calculation.

        Returns
        -------
        
        For each graph in inputs it returns the following:

        conductance_vs_vol: a list of dictionaries
            The length of the list is the number of connected components of the given graph.
            Each element of the list is a dictionary where keys are volumes of clusters and 
            the values are conductance. It can be used to plot the conductance vs volume NCP.

        conductance_vs_size: a list of dictionaries
            The length of the list is the number of connected components of the given graph.
            Each element of the list is a dictionary where keys are sizes of clusters and 
            the values are conductance. It can be used to plot the conductance vs volume NCP.
        """       
        
        return [ncp_algo(inputs[i], ratio=ratio, timeout=timeout, timeout_ncp=timeout_ncp, iterations=iterations, epsilon=epsilon, nthreads=nthreads, multi_threads=multi_threads) for i in range(len(inputs))]

