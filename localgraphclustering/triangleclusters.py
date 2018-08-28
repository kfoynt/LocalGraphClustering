from typing import *
import numpy as np
from .cpp import *
import warnings

def triangleclusters(G):
    """
    TRIANGLECLUSTERS Clustering metrics for clusters of vertex neighborhoods.
    This function studies clusters which are given by vertex neighborhoods.
    Let v be a vertex in a graph, then the cluster associated with v is just
    the set of all neighbors of v and v itself.  We return the clustering
    metrics associated with these clusters for all vertices in the graph.

    Parameters
    ----------

    G: GraphLocal

    Returns
    -------

    cond: Sequence[float]
        conductance of each cluster of a vertex neighborhood

    cut: Sequence[float]
        cut of each cluster

    vol: Sequence[float]
        volume of each cluster

    cc: Sequence[float]
        clustering coefficient of each vertex

    t: Sequence[float]
        number of triangles centered at each vertex
    """
    if G._weighted:
        warnings.warn("The weights of the graph will be discarded.")

    n = G.adjacency_matrix.shape[0]
    return triangleclusters_cpp(n,G.ai,G.aj)
