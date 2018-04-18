from typing import *
import numpy as np
from .algorithms import eig2nL_subgraph

def fiedler_local(G, ref_nodes,
                  timeout: float = 100,
                  iterations: int = 1000,
                  epsilon: float = 1.0e-6):

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
    
        return eig2nL_subgraph(G, ref_nodes, tol_eigs=epsilon)

