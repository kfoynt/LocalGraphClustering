from typing import *
import numpy as np
from .cpp import *
from .GraphLocal import GraphLocal
from .algorithms import sweepcut

def sweep_cut(G: GraphLocal,
              p: Union[Sequence[float],Tuple[Sequence[int],Sequence[float]]],
              do_sort: bool = True,
              normalized: bool = True,
              cpp: bool = True):
    """
    It implements a sweep cut rounding procedure for local graph clustering.

    Parameters
    ----------

    G: GraphLocal

    p: Sequence[float] or Tuple[Sequence[int],Sequence[float]]
        There are three ways to describe the vector used for sweepcut
        The first is just a list of n numbers where n is the nubmer of vertices
        of the graph.

        The second is a pair of vectors that describe a sparse input.

    do_sort: binary
        default = True
        If do_sort is equal to 1 then vector p is sorted in descending order first.
        If do_sort is equal to 0 then vector p is not sorted. In this case,
        only the order of the elements is used. This really only makes
        sense for the sparse input case.

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

    sparsevec = False
    if isinstance(p, tuple):
        if len(p) == 2: # this is a valid sparse input
            nnz_idx = p[0]
            nnz_val = p[1]
            sparsevec = True
        elif len(p) == n:
            pass # this is okay, and will be handled below
        else:
            raise Exception("Unknown input type.")

    if sparsevec == False:
        nnz_idx = np.array(range(0,n), dtype=G.aj.dtype)
        nnz_val = np.array(p, copy=False)
        assert(len(nnz_val) == n)

    nnz_ct = len(nnz_idx)

    if nnz_ct == 0:
        return [[],1]

    if cpp:
        #if fun == None: fun = sweepcut_cpp(G.ai, G.aj, G.lib, 1 - do_sort)
        (length,clus,cond) = sweepcut_cpp(n, G.ai, G.aj, G.adjacency_matrix.data, nnz_idx, nnz_ct, nnz_val, 1 - do_sort)
        return [clus,cond]
    else:
        output = sweepcut(p,G)
        return [output[0],output[1]]
