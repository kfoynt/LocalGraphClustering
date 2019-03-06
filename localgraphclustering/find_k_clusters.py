import scipy as sp
import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from .approximate_PageRank import approximate_PageRank

from .GraphLocal import GraphLocal
from .cpp import *

def compute_embedding(g,
                      node,
                      rho_list,
                      nsamples_from_rho,
                      localmethod,
                      alpha,
                      normalize,
                      normalized_objective,
                      epsilon,
                      iterations):
    
    ref_node = [node]

    sampled_rhos = list(np.geomspace(rho_list[0], rho_list[1], nsamples_from_rho, endpoint=True))

    min_crit = 10000
    min_crit_embedding = 0

    for rho in list(reversed(sampled_rhos)):

        output = approximate_PageRank(g,ref_node,method=localmethod,alpha=alpha,rho=rho,normalize=normalize,normalized_objective=normalized_objective,epsilon=epsilon,iterations=iterations) 

        conductance = g.compute_conductance(output[0])

        crit = conductance
        if crit <= min_crit:
            min_crit = crit
            min_crit_embedding = output
    
    return min_crit_embedding

def find_k_clusters(g, 
                    nclusters, 
                    rho_list, 
                    localmethod: str = 'l1reg-rand', 
                    normalize: bool = False, 
                    normalized_objective: bool = False, 
                    epsilon: float = 1.0e-2, 
                    iterations: int = 10000000,
                    alpha: float = 0.1, 
                    nsamples_from_rho: int = 50, 
                    linkage: str = 'average', 
                    njobs: int = 1, 
                    prefer: str = 'threads', 
                    backend: str = 'multiprocessing',
                    metric: str ='euclidean'):
    """
    Find clusters in a graph using local graph clustering.
    --------------------------------

    This method runs local graph clustering for each node in the graph in parallel.
    Aggregates the embeddings and compute a pairwise distance matrix. 
    Then uses agglomerative clustering to find the clusters. 

    Parameters
    ----------

    G: GraphLocal

    nclusters: int
        Number of clusters to be returned
        
    rho_list: 2D list of floats
        This is an interval of rhos, the regularization parameters for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.

    Parameters (optional)
    ---------------------

    alpha: float
        Default == 0.11
        Teleportation parameter of the personalized PageRank linear system.
        The smaller the more global the personalized PageRank vector is.
        
    nsamples_from_rho: int
        Number of samples of rho parameters to be selected from interval rho_list.

    localmethod: string
        Default = 'l1reg-rand'
        Which method to use.
        Options: 'l1reg', 'l1reg-rand'.
        
    iterations: int
        Default = 1000000
        Maximum number of iterations of ACL algorithm.
        
    epsilon: float
        Default = 1.0e-2
        Tolerance for localmethod

    normalize: bool
        Default = True
        Normalize the output to be directly input into sweepcut routines.
        
    normalized_objective: bool
        Default = True
        Use normalized Laplacian in the objective function, works only for "method=l1reg" and "cpp=True"

    linkage: str
        Default = 'average'
        Which linkage criterion to use for agglomerative clustering. 
        For other options check: 
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html 
        
    metric: str
        Default = 'euclidean'
        Metric for measuring distances among nodes.
        For details check:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        
    njobs: int
        Default = 1
        Number of jobs to be run in parallel
        
    prefer, backend: str
        Check documentation of https://joblib.readthedocs.io/en/latest/


    Returns
    -------

    labels: np.ndarray
    An np.ndarray of the cluster allocation of each node.
    For example labels[i] is the cluster of node i.
    """
    
    n = g._num_vertices
    
    if njobs > 1:
        results = Parallel(n_jobs=njobs, prefer=prefer, backend=backend)(delayed(compute_embedding)(g,node,rho_list,nsamples_from_rho,localmethod,alpha,normalize,normalized_objective,epsilon,iterations) for node in range(n))
    else:
        results =[compute_embedding(g,node,rho_list,nsamples_from_rho,localmethod,alpha,normalize,normalized_objective,epsilon,iterations) for node in range(n)]
        
    sum_ = 0
    JA = [0]
    IA = []
    A  = []
    for data in results:
        vec = data[1]/np.linalg.norm(data[1],2)
        how_many = len(data[0])
        sum_ += how_many
        JA.append(sum_)
        IA.extend(list(data[0]))
        A.extend(list(vec))
    
    X = sp.sparse.csc_matrix((A, IA, JA), shape=(n, n))
    
    X = X.transpose()
    
    Z = pairwise_distances(X, metric=metric, n_jobs=njobs)
    
    clustering = AgglomerativeClustering(n_clusters=nclusters,affinity="precomputed",linkage=linkage).fit(Z)
    labels = clustering.labels_
    
    return labels 

def compute_all_embeddings_and_distances(g, 
                    rho_list, 
                    localmethod: str = 'l1reg-rand', 
                    normalize: bool = False, 
                    normalized_objective: bool = False, 
                    epsilon: float = 1.0e-2, 
                    iterations: int = 10000000,
                    alpha: float = 0.1, 
                    nsamples_from_rho: int = 50, 
                    njobs: int = 1, 
                    prefer: str = 'threads', 
                    backend: str = 'multiprocessing',
                    metric: str ='euclidean'):
    """
    Find clusters in a graph using local graph clustering.
    --------------------------------

    This method runs local graph clustering for each node in the graph in parallel.
    Returns the embeddings for each node in a matrix X. Each row corresponds to an embedding
    of a node. It also returns the pairwise distance matrix Z. For example, component Z[i,j]
    is the distance between nodes i and j.

    Parameters
    ----------

    G: GraphLocal

    rho_list: 2D list of floats
        This is an interval of rhos, the regularization parameters for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.

    Parameters (optional)
    ---------------------

    alpha: float
        Default == 0.11
        Teleportation parameter of the personalized PageRank linear system.
        The smaller the more global the personalized PageRank vector is.
        
    nsamples_from_rho: int
        Number of samples of rho parameters to be selected from interval rho_list.

    localmethod: string
        Default = 'l1reg-rand'
        Which method to use.
        Options: 'l1reg', 'l1reg-rand'.
        
    iterations: int
        Default = 1000000
        Maximum number of iterations of ACL algorithm.
        
    epsilon: float
        Default = 1.0e-2
        Tolerance for localmethod

    normalize: bool
        Default = True
        Normalize the output to be directly input into sweepcut routines.
        
    normalized_objective: bool
        Default = True
        Use normalized Laplacian in the objective function, works only for "method=l1reg" and "cpp=True"

    linkage: str
        Default = 'average'
        Which linkage criterion to use for agglomerative clustering. 
        For other options check: 
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html 
        
    metric: str
        Default = 'euclidean'
        Metric for measuring distances among nodes.
        For details check:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        
    njobs: int
        Default = 1
        Number of jobs to be run in parallel
        
    prefer, backend: str
        Check documentation of https://joblib.readthedocs.io/en/latest/


    Returns
    -------

    X: csc matrix
    The embeddings matrix. Each row corresponds to an embedding of a node. 
    
    Z: 2D np.ndarray
    The pairwise distance matrix Z. For example, component Z[i,j]
    is the distance between nodes i and j.
    """
    
    n = g._num_vertices
    
    if njobs > 1:
        results = Parallel(n_jobs=njobs, prefer=prefer, backend=backend)(delayed(compute_embedding)(g,node,rho_list,nsamples_from_rho,localmethod,alpha,normalize,normalized_objective,epsilon,iterations) for node in range(n))
    else:
        results =[compute_embedding(g,node,rho_list,nsamples_from_rho,localmethod,alpha,normalize,normalized_objective,epsilon,iterations) for node in range(n)]
        
    sum_ = 0
    JA = [0]
    IA = []
    A  = []
    for data in results:
        vec = data[1]/np.linalg.norm(data[1],2)
        how_many = len(data[0])
        sum_ += how_many
        JA.append(sum_)
        IA.extend(list(data[0]))
        A.extend(list(vec))
    
    X = sp.sparse.csc_matrix((A, IA, JA), shape=(n, n))
    
    X = X.transpose()
    
    Z = pairwise_distances(X, metric=metric, n_jobs=njobs)
    
    return X, Z

def compute_k_clusters(nclusters,Z,linkage: str = 'average'):
    """
    Find clusters in a graph using local graph clustering.
    --------------------------------

    This method runs local graph clustering for each node in the graph in parallel.
    Aggregates the embeddings and compute a pairwise distance matrix. 
    Then uses agglomerative clustering to find the clusters. 

    Parameters
    ----------

    nclusters: int
        Number of clusters to be returned
        
    Z: 2D np.ndarray
        The pairwise distance matrix Z. For example, component Z[i,j]
        is the distance between nodes i and j.

    Parameters (optional)
    ---------------------

    linkage: str
        Default = 'average'
        Which linkage criterion to use for agglomerative clustering. 
        For other options check: 
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html 


    Returns
    -------

    labels: np.ndarray
    An np.ndarray of the cluster allocation of each node.
    For example labels[i] is the cluster of node i.
    """
    
    clustering = AgglomerativeClustering(n_clusters=nclusters,affinity="precomputed",linkage=linkage).fit(Z)
    labels = clustering.labels_
    
    return labels 