from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.ista_dinput_dense import ista_dinput_dense

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class L1_regularized_PageRank(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the l1_regularized_PageRank class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input], 
                ref_nodes: Sequence[int],
                timeout: float = 100, 
                iterations: int = 1000,
                alpha: float = 0.15,
                rho: float = 1.0e-6,
                epsilon: float = 1.0e-2) -> Sequence[Output]:
        """
        Computes an l1-regularized PageRank vector. 
        
        Uses the Iterative Soft Thresholding Algorithm (ISTA). This algorithm solves the l1-regularized
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
            Tolerance for ISTA for solving the l1-regularized personalized PageRank problem.
            
        iterations: int
            Default = 100000
            Maximum number of iterations of ISTA algorithm.
                     
        timeout: float
            Default = 100
            Maximum time in seconds. 
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        An np.ndarray (1D embedding) of the nodes for each graph.
        """ 
        
        return [ista_dinput_dense(ref_nodes[i], inputs[i], alpha = alpha, rho = rho, epsilon = epsilon, max_iter = iterations, max_time = timeout) for i in range(len(inputs))]

