from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.page_rank_nibble_algo import page_rank_nibble_algo

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)

class PageRank_nibble(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the pageRank_nibble class.
        """

        super().__init__()

    def produce(self,
                inputs: Sequence[Input], 
                ref_nodes: Sequence[int],
                timeout: float = 100, 
                iterations: Optional[int] = 1000,
                vol: float = 100,
                phi: float = 0.5,
                algorithm: str = 'fista',
                epsilon: float = 1.0e-2,
                cpp: bool = True) -> Sequence[Output]:
        """
        Computes an approximate PageRank vector. Uses the Page Rank Nibble Algorithm. 
        For details please refer to: 
        R. Andersen, F. Chung and K. Lang. Local Graph Partitioning using PageRank Vectors
        link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf
        The algorithm works on the connected component that the given reference node belongs to. 

        Parameters
        ----------

        inputs: Sequence[Graph]

        ref_nodes: Sequence[int]
            A sequence of reference nodes, i.e., nodes of interest around which
            we are looking for a target cluster.
            
        vol: float
             Lower bound for the volume of the output cluster.

        Parameters (optional)
        ---------------------

        phi: float
             Default == 0.5
             Target conductance for the output cluster.

        algorithm: string
            Default == 'fista'
            Algorithm for spectral local graph clustering.
            Options: 'fista', 'ista', 'acl'.
            
        iterations: int
            Default = 100000
            Maximum number of iterations of ACL algorithm.
                     
        timeout: float
            Default = 100
            Maximum time in seconds.
            
        epsilon: float64
            Default == 1.0e-2
            Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.
            
        cpp: boolean
            Default = True
            Use the faster C++ version of FISTA or not.   
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        An np.ndarray (1D embedding) of the nodes for each graph.
        """ 
        
        return [page_rank_nibble_algo(inputs[i], ref_nodes[i], vol = vol, phi = phi, algorithm = algorithm, epsilon = epsilon, max_iter = iterations, max_time = timeout, cpp = cpp) for i in range(len(inputs))]

