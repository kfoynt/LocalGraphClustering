from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.fista_dinput_dense import fista_dinput_dense
from localgraphclustering.proxl1PRaccel import proxl1PRaccel

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class L1_regularized_PageRank_fast(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the l1_regularized_PageRank_fast class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input], 
                p0s: Sequence[Sequence[float]],
                timeout: float = 100, 
                iterations: int = 1000,
                ref_node: int = 0,
                alpha: float = 0.15,
                rho: float = 1.0e-6,
                epsilon: float = 1.0e-6,
                cpp: bool = True
                ) -> Sequence[Output]:
        """
        Computes an l1-regularized PageRank vector. 
        
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

        ref_node: int
            The reference node, i.e., node of interest around which
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
            Default == 1.0e-6
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
        
        if not cpp:
            return [fista_dinput_dense(ref_node, input, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = iterations, max_time = timeout) for input in inputs]
        
        else:
            return [np.abs(proxl1PRaccel(np.uint32(inputs[i].adjacency_matrix.indptr) , np.uint32(inputs[i].adjacency_matrix.indices), inputs[i].adjacency_matrix.data, p0s[i], ref_node, inputs[i].d, inputs[i].d_sqrt, inputs[i].dn_sqrt, alpha = alpha, rho = rho, epsilon = epsilon, maxiter = iterations, max_time = timeout)[2]) for i in range(len(inputs))]
