from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.acl_list import acl_list

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class Approximate_PageRank(GraphBase[Input, Output]):
    """
    Example of a primitive wrapping the ACL algorithm.
    """

    def __init__(self) -> None:
        """
        Initialize the approximate_PageRank class.
        """

        super().__init__()

    def produce(self,
                inputs: Sequence[Input], 
                ref_nodes: Sequence[int],
                timeout: float = 100, 
                iterations: int = 1000,
                alpha: float = 0.15,
                rho: float = 1.0e-6) -> Sequence[Output]:
        """
        Computes an approximate PageRank vector. Uses the Andersen Chung and Lang (ACL) Algorithm. 
        For details please refer to: R. Andersen, F. Chung and K. Lang. Local Graph Partitioning 
        using PageRank Vectors
        link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf. 

        Parameters
        ----------

        inputs: Sequence[Graph]

        ref_nodes: Sequence[int]
            A sequence of reference nodes, i.e., nodes of interest around which
            we are looking for a target cluster.

        Parameters (optional)
        ---------------------

        alpha: float
            Default == 0.15
            Teleportation parameter of the personalized PageRank linear system.
            The smaller the more global the personalized PageRank vector is.
            
        rho: float
            Defaul == 1.0e-6
            Regularization parameter for the l1-norm of the model.
            
        iterations: int
            Default = 1000
            Maximum number of iterations of ACL algorithm.
                     
        timeout: float
            Default = 100
            Maximum time in seconds   
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        An np.ndarray (1D embedding) of the nodes for each graph.
        """ 
            
        return [acl_list(ref_nodes[i], inputs[i], alpha = alpha, rho = rho, max_iter = iterations, max_time = timeout) for i in range(len(inputs))]
