from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.eig2nL_subgraph import eig2nL_subgraph

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class Spectral_partitioning_subgraph(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the Spectral_partitioning_subgraph class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input],
                ref_nodes: Sequence[int],
                timeout: float = 100,
                iterations: int = 1000,
                epsilon: float = 1.0e-6) -> Sequence[Output]:
        """
        Computes the eigenvector that corresponds to the second smallest eigenvalue 
        of the normalized Laplacian matrix for a subgraph that corresponds to a given set of nodes.

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
    
        return [eig2nL_subgraph(inputs[i], ref_nodes[i], tol_eigs=epsilon) for i in range(len(inputs))]

