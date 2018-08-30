from typing import *
import numpy as np
from .GraphLocal import GraphLocal
from .algorithms import eig2_nL
from .algorithms import eig2nL_subgraph

def fiedler(G: GraphLocal, epsilon: float = 1.0e-6, normalize: bool = True):
        """
        Computes the eigenvector that corresponds to the second smallest eigenvalue
        of the normalized Laplacian matrix.

        Parameters (mandatory)
        ----------------------

        G: GraphLocal

        Parameters (optional)
        ---------------------

        epsilon: float
            Default == 1.0e-6
            Tolerance for computation of the eigenvector that corresponds to
            the second smallest eigenvalue of the normalized Laplacian matrix.

        normalize: bool
            Default = True
            Normalize the output to be directly input into sweepcut routines.

        Returns
        -------

        p: ndarray
            Eigenvector that corresponds to the second smallest eigenvalue of the
            normalized Laplacian matrix.
        """

        return eig2_nL(G, tol_eigs=epsilon, normalize=normalize)

def fiedler_local(G, ref_nodes,
                  epsilon: float = 1.0e-6,
                  normalize: bool = True):
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

        normalize: bool
            Default = True
            Normalize the output to be directly input into sweepcut routines.

        Returns
        -------

        For each input graph it computes the following:

        p: ndarray
            Eigenvector that corresponds to the second smallest eigenvalue of the
            normalized Laplacian matrix.
        """

        return eig2nL_subgraph(G, ref_nodes, tol_eigs=epsilon, normalize=normalize)
