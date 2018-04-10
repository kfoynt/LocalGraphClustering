from typing import *
import numpy as np
from localgraphclustering.cpp.sweepcut_cpp import sweepcut_cpp
from localgraphclustering.graph_class_local import graph_class_local
from localgraphclustering.algorithms.sweepcut import sweepcut

def sweep_cut(G: graph_class_local, 
              p: Sequence[float],
              do_sort: bool = True,
              normalized: bool = True,
              cpp: bool = True):
    """
    It implements a sweep cut rounding procedure for local graph clustering. 

    Parameters
    ----------

    G: graph_class_local

    p: Sequence[float]
        A vector that is used to perform rounding.
           
    do_sort: binary 
        default = True 
        If do_sort is equal to 1 then vector p is sorted in descending order first.
        If do_sort is equal to 0 then vector p is not sorted.

    normalized: binary
        default = True
        If it is true, p will be normalized with respect to the degree of each nonzero index in G.

    cpp: bool
        default = True
        Use the faster C++ version or not.
            
    Returns
    -------
        
    It returns in a list of length 2 with the following:
        
    output 0: list
        Stores indices of the best clusters found by the last called rounding procedure.
           
    output 1: float
        Stores the value of the best conductance found by the last called rounding procedure.
    """ 
       
    n = G.adjacency_matrix.shape[0]

    p = np.array(p)

    nnz_idx = p.nonzero()[0]
    nnz_ct = nnz_idx.shape[0]

    if nnz_ct == 0:
        return [[],1]

    sc_p = np.zeros(nnz_ct)

    if normalized:
        for i in range(nnz_ct):
            degree = G.d[nnz_idx[i]]
            sc_p[i] = p[nnz_idx[i]]/degree
        p[nnz_idx] = sc_p
    else:
        for i in range(nnz_ct):
            sc_p[i] = p[nnz_idx[i]]   
    
    if cpp:
        (length,clus,cond) = sweepcut_cpp(n, np.uint32(G.adjacency_matrix.indptr), 
                                          np.uint32(G.adjacency_matrix.indices), 
                                          G.adjacency_matrix.data, nnz_idx, nnz_ct, 
                                          sc_p, 1 - do_sort,G.lib)
        return [clus,cond]
    else:
        output = sweepcut(p,G)
        return [output[0],output[1]]


