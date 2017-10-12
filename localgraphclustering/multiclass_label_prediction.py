from typing import *
import numpy as np
from scipy import sparse as sp
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.fista_dinput_dense import fista_dinput_dense
from localgraphclustering.proxl1PRaccel import proxl1PRaccel
from localgraphclustering.multiclass_label_prediction_algo import multiclass_label_prediction_algo

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class Multiclass_label_prediction(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the Multiclass_label_prediction class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input], 
                timeout: float = 100, 
                iterations: int = 1000,
                labels: np.ndarray = [],
                alpha: float = 0.15,
                rho: float = 1.0e-6,
                epsilon: float = 1.0e-2,
                cpp: bool = True) -> Sequence[Output]:
        """
        This function predicts labels for unlabelled nodes. For details refer to:
        D. Gleich and M. Mahoney. Variational 
        Using Local Spectral Methods to Robustify Graph-Based Learning Algorithms. SIGKDD 2015.
        https://www.stat.berkeley.edu/~mmahoney/pubs/robustifying-kdd15.pdf

        Parameters (mandatory)
        ----------------------
        
        inputs: Sequence[Graph]

        labels: list of lists
            Each list of this list corresponds to indices of nodes that are assumed to belong in
            a certain class. For example, list[i] is a list of indices of nodes that are assumed to 
            belong in class i.

        Parameters (optional)
        ---------------------

        alpha: float, double
            Default == 0.15
            Teleportation parameter of the personalized PageRank linear system.
            The smaller the more global the personalized PageRank vector is.

        rho: float, double
            Defaul == 1.0e-10
            Regularization parameter for the l1-norm of the model.

        epsilon: float, double
            Default == 1.0e-2
            Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.

        iterations: int
            Default = 100000
            Maximum number of iterations of FISTA algorithm.
                     
        timeout: float
            Default = 100
            Maximum time in seconds.

        cpp: bool
            default = True
            Use the faster C++ version of FISTA or not.

           Returns
           -------
           
           For each graph in inputs it returns in a list of length 3 the following:

           output 0: list of indices that holds the class for each node.
               For example classes[i] is the class of node i.

           output 1: list of lists. Each componenent of the list is a list that holds the rank
               of the nodes for each class. For details see [1].

           output 2: a list of numpy arrays. Each array in this list corresponds to the diffusion vector
               returned by personalized PageRank for each rank. For details see [1].

           [1] D. Gleich and M. Mahoney. Variational 
           Using Local Spectral Methods to Robustify Graph-Based Learning Algorithms. SIGKDD 2015.
           https://www.stat.berkeley.edu/~mmahoney/pubs/robustifying-kdd15.pdf
        """   

        return [multiclass_label_prediction_algo(labels=labels, g=input, alpha=alpha, rho=rho, epsilon=epsilon, max_iter=iterations, max_time=timeout, cpp=True) for input in inputs]