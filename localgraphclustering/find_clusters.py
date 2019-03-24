import scipy as sp
import numpy as np
import time
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from .approximate_PageRank import approximate_PageRank
from .approximate_PageRank_weighted import approximate_PageRank_weighted
from .flow_clustering import flow_clustering

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

#     if not is_weighted:
    
    for rho in list(reversed(sampled_rhos)):

        output = approximate_PageRank(g,ref_node,method=localmethod,alpha=alpha,rho=rho,normalize=normalize,normalized_objective=normalized_objective,epsilon=epsilon,iterations=iterations) 


        conductance = g.compute_conductance(output[0])

        crit = conductance
        if crit <= min_crit:
            min_crit = crit
            min_crit_embedding = output
                
#     else:
#         rho = rho_list[0]
#         min_crit_embedding = approximate_PageRank_weighted(g,ref_node,alpha=alpha,rho=rho)
    
    return min_crit_embedding

def compute_embedding_and_improve(g,
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

#     if not is_weighted:
    
    for rho in list(reversed(sampled_rhos)):

        output = approximate_PageRank(g,ref_node,method=localmethod,alpha=alpha,rho=rho,normalize=normalize,normalized_objective=normalized_objective,epsilon=epsilon,iterations=iterations) 


        conductance = g.compute_conductance(output[0])

        crit = conductance
        if crit <= min_crit:
            min_crit = crit
            min_crit_embedding = output
                
#     else:
#         rho = rho_list[0]
#         min_crit_embedding = approximate_PageRank_weighted(g,ref_node,alpha=alpha,rho=rho)
    
    output_mqi = flow_clustering(g,min_crit_embedding[0],method="mqi_weighted")
    
    return output_mqi

def find_clusters(g, 
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

    g: GraphLocal

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
    
#     is_weighted = g._weighted
    
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
    This method runs local graph clustering for each node in the graph in parallel.
    Returns the embeddings for each node in a matrix X. Each row corresponds to an embedding
    of a node. It also returns the pairwise distance matrix Z. For example, component Z[i,j]
    is the distance between nodes i and j.

    Parameters
    ----------

    g: GraphLocal

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
    
#     is_weighted = g._weighted
    
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
    
    X = X.transpose().tocsr()
    
    Z = pairwise_distances(X, metric=metric, n_jobs=njobs)
    
    return X, Z

def compute_clusters_given_distance(nclusters,Z,linkage: str = 'average'):
    """
    Find clusters in a graph using local graph clustering.
    --------------------------------

    Each node is represented by a sparse local graph clustering vector.
    Then it uses agglomerative clustering to find the clusters. 

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

def graph_segmentation(g, 
                    rho_list, 
                    localmethod: str = 'l1reg-rand', 
                    normalize: bool = False, 
                    normalized_objective: bool = False, 
                    epsilon: float = 1.0e-2, 
                    iterations: int = 10000000,
                    alpha: float = 0.1, 
                    nsamples_from_rho: int = 50,
                    njobs = 1,
                    prefer: str = 'threads', 
                    backend: str = 'multiprocessing',
                    how_many_in_parallel = 5,
                    ratio = 0.01):
    """
    Segment the graph into pieces by peeling off clusters in parallel using local graph clustering.
    --------------------------------

    Parameters
    ----------

    g: GraphLocal

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
        
    njobs: int
        Default = 1
        Number of jobs to be run in parallel
        
    prefer, backend: str
        Check documentation of https://joblib.readthedocs.io/en/latest/
        
    how_many_in_parallel: int
        Default = 20
        Number of segments that are computed in parallel. 
        There is a trade-off here.    
        
    ratio: float
        Default = 0.01
        Let n be the number of nodes, this segmentation code will ignore the last ratio*n nodes,
        and it will cluster them as one cluster.

    Returns
    -------

    info: list of lists
    Each element of the list is another list with two elements.
    The first element is the indices of the a segment, while the second element
    is the vector representation of that segment.
    
    labels: np.ndarray
    An np.ndarray of the cluster allocation of each node.
    For example labels[i] is the cluster of node i.
    
    """
    
    g_copy = GraphLocal()
    g_copy.from_sparse_adjacency(g.adjacency_matrix)
    candidates = list(range(g_copy._num_vertices))

    labels = np.zeros(g_copy._num_vertices,dtype=np.int32)
    info = []
    ct = 0

    while True:
        
        if njobs > 1:
            select_from = list(range(g_copy._num_vertices))
            ref_nodes = random.sample(select_from, min(how_many_in_parallel,len(select_from)))
            
            results = Parallel(n_jobs=njobs, prefer=prefer, backend=backend)(delayed(compute_embedding)(g_copy,node,rho_list,nsamples_from_rho,localmethod,alpha,normalize,normalized_objective,epsilon,iterations) for node in ref_nodes)
        else:
            select_from = list(range(g_copy._num_vertices))
            ref_nodes = random.sample(select_from, njobs)
            
            results =[compute_embedding(g_copy,node,rho_list,nsamples_from_rho,localmethod,alpha,normalize,normalized_objective,epsilon,iterations) for node in ref_nodes]
    
        union_sets_to_remove = set()
        for res in results:
            idx = [candidates[i] for i in res[0]]
            labels[idx] = ct
            ct += 1
            union_sets_to_remove.update(res[0])
            info.append([idx,res[1]])
    
        for index in sorted(list(union_sets_to_remove), reverse=True):
            del candidates[index]
    
        indices = list(set(range(g_copy._num_vertices)) - set(union_sets_to_remove))
        A = g_copy.adjacency_matrix.tocsr()[indices, :].tocsc()[:, indices]

        g_copy = GraphLocal()
        g_copy.from_sparse_adjacency(A)
        
        print ("Percentage completed: ", 100-len(candidates)/g._num_vertices*100, end="\r")
        
        if len(candidates) <= g._num_vertices*ratio:
            for i in candidates:
                labels[i] = ct
                ct += 1
            return labels, info
        
        
def graph_segmentation_with_improve(g, 
                    rho_list, 
                    localmethod: str = 'l1reg-rand', 
                    normalize: bool = False, 
                    normalized_objective: bool = False, 
                    epsilon: float = 1.0e-2, 
                    iterations: int = 10000000,
                    alpha: float = 0.1, 
                    nsamples_from_rho: int = 50,
                    njobs = 1,
                    prefer: str = 'threads', 
                    backend: str = 'multiprocessing',
                    how_many_in_parallel = 5,
                    ratio = 0.01):
    """
    Segment the graph into pieces by peeling off clusters in parallel using local graph clustering.
    --------------------------------

    Parameters
    ----------

    g: GraphLocal

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
        
    njobs: int
        Default = 1
        Number of jobs to be run in parallel
        
    prefer, backend: str
        Check documentation of https://joblib.readthedocs.io/en/latest/
        
    how_many_in_parallel: int
        Default = 20
        Number of segments that are computed in parallel. 
        There is a trade-off here.    
        
    ratio: float
        Default = 0.01
        Let n be the number of nodes, this segmentation code will ignore the last ratio*n nodes,
        and it will cluster them as one cluster.

    Returns
    -------

    info: list of lists
    Each element of the list is another list with two elements.
    The first element is the indices of the a segment, while the second element
    is the vector representation of that segment.
    
    labels: np.ndarray
    An np.ndarray of the cluster allocation of each node.
    For example labels[i] is the cluster of node i.
    
    """
    
    g_copy = GraphLocal()
    g_copy.from_sparse_adjacency(g.adjacency_matrix)
    candidates = list(range(g_copy._num_vertices))

    labels = np.zeros(g_copy._num_vertices,dtype=np.int32)
    info = []
    ct = 0

    while True:
        
        if njobs > 1:
            select_from = list(range(g_copy._num_vertices))
            ref_nodes = random.sample(select_from, min(how_many_in_parallel,len(select_from)))
            
            results = Parallel(n_jobs=njobs, prefer=prefer, backend=backend)(delayed(compute_embedding_and_improve)(g_copy,node,rho_list,nsamples_from_rho,localmethod,alpha,normalize,normalized_objective,epsilon,iterations) for node in ref_nodes)
        else:
            select_from = list(range(g_copy._num_vertices))
            ref_nodes = random.sample(select_from, njobs)
            
            results =[compute_embedding_and_improve(g_copy,node,rho_list,nsamples_from_rho,localmethod,alpha,normalize,normalized_objective,epsilon,iterations) for node in ref_nodes]
    
        union_sets_to_remove = set()
        for res in results:
            idx = [candidates[i] for i in res[0]]
            labels[idx] = ct
            ct += 1
            union_sets_to_remove.update(res[0])
            info.append([idx,res[1]])
    
        for index in sorted(list(union_sets_to_remove), reverse=True):
            del candidates[index]
    
        indices = list(set(range(g_copy._num_vertices)) - set(union_sets_to_remove))
        A = g_copy.adjacency_matrix.tocsr()[indices, :].tocsc()[:, indices]

        g_copy = GraphLocal()
        g_copy.from_sparse_adjacency(A)
        
        print ("Percentage completed: ", 100-len(candidates)/g._num_vertices*100, end="\r")
        
        if len(candidates) <= g._num_vertices*ratio:
            for i in candidates:
                labels[i] = ct
                ct += 1
            return labels, info
        
def compute_embeddings_and_distances_from_region_adjacency(g,info, metric='euclidean', n_jobs=1):
    """
    This method runs local graph clustering for each node in the region adjacency graph.
    Returns the embeddings for each node in a matrix X. Each row corresponds to an embedding
    of a node in the region adjacency graph. It also returns the pairwise distance matrix Z. 
    For example, component Z[i,j] is the distance between nodes i and j.

    Parameters
    ----------
    
    g: GraphLocal

    info: list of lists
    Each element of the list is another list with two elements.
    The first element is the indices of the a segment, while the second element
    is the vector representation of that segment.

    Parameters (optional)
    ---------------------
        
    metric: str
        Default = 'euclidean'
        Metric for measuring distances among nodes.
        For details check:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
        
    njobs: int
        Default = 1
        Number of jobs to be run in parallel

    Returns
    -------

    X: csc matrix
    The embeddings matrix. Each row corresponds to an embedding of a node in the regiona adjacency graph. 
    
    Z: 2D np.ndarray
    The pairwise distance matrix Z. For example, component Z[i,j]
    is the distance between nodes i and j.
    """
    
    sum_ = 0
    JA = [0]
    IA = []
    A  = []

    for data in info:
        vec = data[1]/np.linalg.norm(data[1],2)
        how_many = len(data[0])
        sum_ += how_many
        JA.append(sum_)
        IA.extend(list(data[0]))
        A.extend(list(vec))

    X = sp.sparse.csc_matrix((A, IA, JA), shape=(g._num_vertices, len(info)))

    X = X.transpose()
    
    Z = pairwise_distances(X, metric='euclidean', n_jobs=6)
    
    return X, Z
    
def compute_clusters_from_region_adjacency(g,nclusters,Z,info,linkage: str = 'complete'):
    """
    Find clusters in a graph using a region adjacency graph.
    --------------------------------

    Each node represents a segment in the original graph. 
    Each segment is represented by a sparse local graph clustering vector.
    Then it uses agglomerative clustering to find the clusters. 

    Parameters
    ----------
    
    g: GraphLocal

    nclusters: int
        Number of clusters to be returned
        
    Z: 2D np.ndarray
        The pairwise distance matrix Z. For example, component Z[i,j]
        is the distance between nodes i and j.
        
    info: list of lists
    Each element of the list is another list with two elements.
    The first element is the indices of the a segment, while the second element
    is the vector representation of that segment.

    Parameters (optional)
    ---------------------

    linkage: str
        Default = 'complete'
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
    
    expanded_labels = np.zeros(g._num_vertices, dtype=int)
    for i in range(len(labels)):
        for j in info[i][0]:
            expanded_labels[j] = labels[i]
    
    return expanded_labels 