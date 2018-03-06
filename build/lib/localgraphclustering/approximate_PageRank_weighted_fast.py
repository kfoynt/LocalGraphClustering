from typing import *
from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.acl_list import acl_list
from localgraphclustering.aclpagerank_weighted_cpp import aclpagerank_weighted_cpp

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class Approximate_PageRank_weighted_fast(GraphBase[Input, Output]):
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
                iterations: int = 1000,
                alpha: float = 0.15,
                rho: float = 1.0e-6,
                xlength: int = 1000) -> Sequence[Output]:
        """
        Computes an approximate PageRank vector on a weighted graph. Uses the modified Andersen Chung and Lang (ACL) Algorithm. 
        Now, the diffusion from one node to its neighbors is proportional to the edge weight.
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

        xlength: int
            Default = 1000
            Maximum number of node ids in the solution vector

        iterations: int
            Default = 1000
            Maximum number of iterations of ACL algorithm.
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        An np.ndarray (2D embedding) of the nodes for each graph.

        The first dimension represents node ids where the pagerank value is nonzero,
        and the second dimension represents the corresponding pagerank value. Outputs 
        are sorted in the descending order of pagerank values.
        """ 
            
        output = [[] for i in range(len(inputs))]

        counter = 0

        for i in range(len(inputs)):

            n = inputs[i].adjacency_matrix.shape[0]

            (actual_length,actual_xids,actual_values) = aclpagerank_weighted_cpp(n,np.uint32(inputs[i].adjacency_matrix.indptr),
                np.uint32(inputs[i].adjacency_matrix.indices),np.uint32(inputs[i].adjacency_matrix.data),alpha,rho,
                [ref_nodes[i]],1,iterations,inputs[i].lib,xlength=xlength)
            
            output[counter] = [actual_xids,actual_values]

            counter += 1

        return output
