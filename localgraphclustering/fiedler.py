from typing import *
import numpy as np
from localgraphclustering.graph_class_local import graph_class_local
from localgraphclustering.algorithms.eig2_nL import eig2_nL

def fiedler(G: graph_class_local, epsilon: float = 1.0e-6):
        """
        Computes the eigenvector that corresponds to the second smallest eigenvalue 
        of the normalized Laplacian matrix.

        Parameters (mandatory)
        ----------------------

        G: graph_class_local

        Parameters (optional)
        ---------------------

        epsilon: float
            Default == 1.0e-6
            Tolerance for computation of the eigenvector that corresponds to 
            the second smallest eigenvalue of the normalized Laplacian matrix.

        Returns
        -------

        p: ndarray
            Eigenvector that corresponds to the second smallest eigenvalue of the 
            normalized Laplacian matrix.
        """ 
    
        return eig2_nL(G, tol_eigs=epsilon)
