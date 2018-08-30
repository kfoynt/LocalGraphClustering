import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg


def eig2_nL(g, tol_eigs = 1.0e-6, normalize:bool = True):
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

    emb_eig_val, p = splinalg.eigsh(L, which='SM', k=2, tol = tol_eigs)

    f = np.real(p[:,1])
    if normalize:
        f *= g.dn_sqrt
    return f, emb_eig_val

"""
Random walks and local cuts in graphs, Chung, LAA 2007
We just form the sub-matrix of the Laplacian and use the eigenvector there.
"""
def eig2nL_subgraph(g, ref_nodes, tol_eigs = 1.0e-6, normalize: bool = True):

    # TODO, optimize this routine to avoid creating the full L
    n = g.adjacency_matrix.shape[0]
    D_sqrt_neg = sp.sparse.spdiags(g.dn_sqrt.transpose(), 0, n, n)

    L = sp.sparse.identity(n) - D_sqrt_neg.dot((g.adjacency_matrix.dot(D_sqrt_neg)))

    L_sub = g.adjacency_matrix.tocsr()[ref_nodes, :].tocsc()[:, ref_nodes]

    emb_eig_val, emb_eig = splinalg.eigsh(L_sub, which='SM', k=1, tol=tol_eigs)
    f = np.real(emb_eig[:,0])
    if normalize:
        f *= g.dn_sqrt[ref_nodes]
    return ((ref_nodes,f), emb_eig_val)
