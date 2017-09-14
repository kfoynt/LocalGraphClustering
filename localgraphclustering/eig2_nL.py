import numpy as np
import scipy.sparse.linalg as splinalg
from localgraphclustering.sweepCut import *

def eig2_nL(g, tol_eigs = 1.0e-6):
    """
       DESCRIPTION
       -----------
       
       Computes the eigenvector that corresponds to the second smallest eigenvalue 
       of the normalized Laplacian matrix then it uses sweep cut to round the solution.
       
       PARAMETERS (mandatory)
       ----------------------
   
       g:         graph object
       
       PARAMETERS (optional)
       ---------------------
      
       tol_eigs: positive float, double
                 default == 1.0e-6
                 Tolerance for computation of the eigenvector that corresponds to 
                 the second smallest eigenvalue of the normalized Laplacian matrix.
       
       RETURNS
       ------
       
       p:    csr_matrix, float
             Eigenvector that corresponds to the second smallest eigenvalue of the 
             normalized Laplacian matrix.
       
       best_cluster: list
                     A list of nodes that correspond to the cluster with the best 
                     conductance among the one that were found by this method.
                      
       best_conductance: float
                         Conductance value that corresponds to the cluster with the best 
                         conductance that was found by this method.
                          
       sweep_profile: list of objects
                      A two dimensional list of objects. For example,
                      sweep_profile[0] contains a numpy array with all conductances for all
                      clusters that were calculated by sweep_cut.
                      sweep_profile[1] is a multidimensional list that contains the indices
                      of all clusters that were calculated by sweep_cut. For example,
                      sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                      that was calculated by sweep_cut. The number of clusters is unknwon apriori 
                      at depends on the data and that parameter setting of this method.
    """
    
    n = g.A.shape[0]
    
    D_sqrt_neg = sp.spdiags(g.dn_sqrt.transpose(), 0, n, n)
    
    L = sp.identity(n) - D_sqrt_neg.dot((g.A.dot(D_sqrt_neg)))
    
    emb_eig_val, p = splinalg.eigs(L, which='SM', k=2, tol = tol_eigs)

    p = np.real(p[:,1])
    
    sweep_eig2nL = sweepCut()
    sweep_eig2nL.sweep_general(p,g)
    
    return p, sweep_eig2nL.best_cluster, sweep_eig2nL.best_conductance, sweep_eig2nL.sweep_profile
    
