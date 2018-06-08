import numpy as np

def neighborhoodmin(G,vals,strict):
    """
    Find extrema in a graph based on neighborhoods.
    Parameters
    ----------
    G: GraphLocal
    vals: Sequence[float]
        features of neighborhoods used to compare against each other, i.e. conductance
    strict: bool
        If True, find a set of vertices where vals(i) < vals(j) for all neighbors N(j)
        i.e. local minima in the space of the graph
        If False, find a set of vertices where vals(i) <= vals(j) for all neighbors N(j)
        i.e. local minima in the space of the graph
    Returns
    -------
    minverts: Sequence[int]
        the set of vertices
    minvals: Sequence[float]
        the set of min values
    """
    n = G.adjacency_matrix.shape[0]
    minverts = []
    ai = np.uint32(G.adjacency_matrix.indptr)
    aj = np.uint32(G.adjacency_matrix.indices)
    for i in range(n):
        neigh = aj[ai[i]:ai[i+1]]
        neighmin = min(vals[neigh])
        if strict:
            if vals[i] < neighmin:
                minverts.append(i)
        else:
            if vals[i] <= neighmin:
                minverts.append(i)
    minvals = vals[minverts]

    return minverts, minvals