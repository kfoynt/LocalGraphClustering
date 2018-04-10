import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg

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
    """
    
    n = g.adjacency_matrix.shape[0]
    
    D_sqrt_neg = sp.sparse.spdiags(g.dn_sqrt.transpose(), 0, n, n)
    
    L = sp.sparse.identity(n) - D_sqrt_neg.dot((g.adjacency_matrix.dot(D_sqrt_neg)))
    
    emb_eig_val, p = splinalg.eigs(L, which='SM', k=2, tol = tol_eigs)

    return np.real(p[:,1])
