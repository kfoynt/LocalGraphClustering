import numpy as np
from scipy import sparse as sp
from scipy import linalg as sp_linalg

def eig2L_subgraph(A, ref_nodes):
    
    A_sub = A.tocsr()[ref_nodes, :].tocsc()[:, ref_nodes]
    
    n = A_sub.shape[0]
    
    d_sqrt = np.zeros(n)
    
    keep_nodes = []
    
    for i in range(n):
        d_sqrt[i] = np.sqrt(A_sub[i,:].nonzero()[1].shape[0])
        if d_sqrt[i] > 0:
            keep_nodes.append(i)
    
    A_sub = A_sub.tocsr()[keep_nodes, :].tocsc()[:, keep_nodes]
    
    d_sqrt = d_sqrt[keep_nodes]

    n = len(keep_nodes)
    
    d_sqrt_neg = np.zeros((n,1))      
        
    for i in xrange(n):
        d_sqrt_neg[i] = 1/d_sqrt[i]        
    
    D_sqrt_neg = sp.spdiags(d_sqrt_neg.transpose(), 0, n, n)
    
    L_sub = sp.identity(n) - D_sqrt_neg.dot((A_sub.dot(D_sqrt_neg)))
    
    if 2 >= A_sub.shape[0]-1:
        return [], [1]
    else:
        emb_eig_val, emb_eig = sp.linalg.eigs(L_sub, which='SM', k=2, tol=1.0e-6)
        return emb_eig[:,1], emb_eig_val[1]