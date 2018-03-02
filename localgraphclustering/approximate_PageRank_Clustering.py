from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.aclpagerank_cpp import aclpagerank_cpp
from localgraphclustering import sweepCut_fast

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class Approximate_PageRank_Clustering(GraphBase[Input, Output]):
    """
    Approximate_PageRank + Rounding algorithm, returns an cluster as an output.
    """

    def __init__(self) -> None:
        """
        Initialize the Approximate_PageRank_Clustering class.
        """

        super().__init__()

    def produce(self,
                inputs: Sequence[Input], 
                ref_nodes: Sequence[int],
                iterations: int = 100000,
                alpha: float = 0.15,
                rho: float = 1.0e-6,
                xlength: int = 10000) -> Sequence[Output]:
        """
        Computes an approximate PageRank vector. Uses the Andersen Chung and Lang (ACL) Algorithm. 
        For details please refer to: R. Andersen, F. Chung and K. Lang. Local Graph Partitioning 
        using PageRank Vectors
        link: http://www.cs.cmu.edu/afs/cs/user/glmiller/public/Scientific-Computing/F-11/RelatedWork/local_partitioning_full.pdf. 
        Then rounds the output of approximate PageRank to return a list of nodes. 
        
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
            Default = 10000
            Maximum number of node ids in the solution vector

        iterations: int
            Default = 100000
            Maximum number of iterations of ACL algorithm.
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        A list of nodes which corresponds to a cluster.
        """ 
            
        output = [[] for i in range(len(inputs))]
        
        sc_fast = sweepCut_fast.SweepCut_fast()

        counter = 0

        for i in range(len(inputs)):

            n = inputs[i].adjacency_matrix.shape[0]

            (actual_length,actual_xids,actual_values) = aclpagerank_cpp(n,np.uint32(inputs[i].adjacency_matrix.indptr),
                np.uint32(inputs[i].adjacency_matrix.indices),alpha,rho,[ref_nodes[i]],1,iterations,inputs[i].lib,xlength=xlength)

            output_sc_fast = sc_fast.produce([inputs[i]],p=actual_values)

            # Get the rounded solution
            output[counter] = actual_xids[output_sc_fast[0][0]]

            counter += 1

        return output

