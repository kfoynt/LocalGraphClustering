from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering import l1_regularized_PageRank_fast
from localgraphclustering import sweepCut_fast

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class L1_regularized_PageRank_Clustering(GraphBase[Input, Output]):
    """
    L1 regularized PageRank + Rounding algorithm, returns an cluster as an output.
    """

    def __init__(self) -> None:
        """
        Initialize the L1_regularized_PageRank_Clustering class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input], 
                ref_nodes: Sequence[int],
                timeout: float = 100, 
                iterations: int = 1000,
                alpha: float = 0.15,
                rho: float = 1.0e-6,
                epsilon: float = 1.0e-2,
                cpp: bool = True
                ) -> Sequence[Output]:
        """
        Computes an l1-regularized PageRank vector and rounds the vector to produce a cluster.
        
        Uses the Fast Iterative Soft Thresholding Algorithm (FISTA). This algorithm solves the l1-regularized
        personalized PageRank problem.

        The l1-regularized personalized PageRank problem is defined as

        min rho*||p||_1 + <c,p> + <p,Q*p>

        where p is the PageRank vector, ||p||_1 is the l1-norm of p, rho is the regularization parameter 
        of the l1-norm, c is the right hand side of the personalized PageRank linear system and Q is the 
        symmetrized personalized PageRank matrix.

        For details please refer to: 
        K. Fountoulakis, F. Roosta-Khorasani, J. Shun, X. Cheng and M. Mahoney. Variational 
        Perspective on Local Graph Clustering. arXiv:1602.01886, 2017.
        arXiv link:https://arxiv.org/abs/1602.01886

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
            Defaul == 1.0e-5
            Regularization parameter for the l1-norm of the model.
            
        epsilon: float64
            Default == 1.0e-2
            Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.
            
        iterations: int
            Default = 100000
            Maximum number of iterations of FISTA algorithm.
                     
        timeout: float
            Default = 100
            Maximum time in seconds.
            
        cpp: boolean
            Default = True
            Use the faster C++ version of FISTA or not.            
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        An np.ndarray (1D embedding) of the nodes for each graph.
        """ 
            
        output = [[] for i in range(len(inputs))]

        counter = 0
        
        l1reg_fast = l1_regularized_PageRank_fast.L1_regularized_PageRank_fast()
        sc_fast = sweepCut_fast.SweepCut_fast()

        for i in range(len(inputs)):

            n = inputs[i].adjacency_matrix.shape[0]

            output_l1reg_fast = l1reg_fast.produce([inputs[i]],[ref_nodes[i]],alpha=alpha,rho=rho,epsilon=epsilon,iterations=iterations,timeout=timeout,cpp=cpp)[0]
            
            output[counter] = sc_fast.produce([inputs[i]],p=output_l1reg_fast)[0][0]

            counter += 1

        return output


