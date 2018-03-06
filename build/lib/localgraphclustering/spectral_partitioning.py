from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.eig2_nL import eig2_nL

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class Spectral_partitioning(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the Spectral_partitioning class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input],
                timeout: float = 100,
                iterations: int = 1000,
                epsilon: float = 1.0e-6) -> Sequence[Output]:
        """
        Computes the eigenvector that corresponds to the second smallest eigenvalue 
        of the normalized Laplacian matrix.

        Parameters (mandatory)
        ----------------------

        inputs: Sequence[Graph]

        Parameters (optional)
        ---------------------

        epsilon: float
            Default == 1.0e-6
            Tolerance for computation of the eigenvector that corresponds to 
            the second smallest eigenvalue of the normalized Laplacian matrix.

        iterations: not used

        timeout: not used

        Returns
        -------

        For each input graph it computes the following:

        p: ndarray
            Eigenvector that corresponds to the second smallest eigenvalue of the 
            normalized Laplacian matrix.
        """ 
    
        return [eig2_nL(input, tol_eigs=epsilon) for input in inputs]
