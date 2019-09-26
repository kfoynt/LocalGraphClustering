import scipy as sp
import numpy as np
import time
import random
import queue
import multiprocessing as mp
import copy
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from .approximate_PageRank import approximate_PageRank
from .approximate_PageRank_weighted import approximate_PageRank_weighted
from .spectral_clustering import spectral_clustering
from .flow_clustering import flow_clustering

from .GraphLocal import GraphLocal
from .cpp import *

def compute_embedding(g,
                      node,
                      rho_list,
                      alpha_list,
                      nsamples_from_rho,
                      nsamples_from_alpha,
                      localmethod,
                      normalize,
                      normalized_objective,
                      epsilon,
                      iterations,
                      cpp):
    
    ref_node = [node]

    sampled_rhos = list(np.geomspace(rho_list[0], rho_list[1], nsamples_from_rho, endpoint=True))
    
    sampled_alphas = list(np.geomspace(alpha_list[0], alpha_list[1], nsamples_from_alpha, endpoint=True))

    min_crit = 10000
    min_crit_embedding = 0
    
    for alpha in list(reversed(sampled_alphas)):
    
        for rho in list(reversed(sampled_rhos)):

            output = approximate_PageRank(g,ref_node,cpp=cpp,method=localmethod,alpha=alpha,rho=rho,normalize=normalize,normalized_objective=normalized_objective,epsilon=epsilon,iterations=iterations) 

            conductance = g.compute_conductance(output[0])

            crit = conductance
            if crit <= min_crit:
                min_crit = crit
                min_crit_embedding = output
    
    return min_crit_embedding

def compute_embedding_and_improve(g,
                      node,
                      rho_list,
                      alpha_list,
                      nsamples_from_rho,
                      nsamples_from_alpha,
                      localmethod,
                      normalize,
                      normalized_objective,
                      epsilon,
                      iterations,
                      cpp):
    
    ref_node = [node]

    sampled_rhos = list(np.geomspace(rho_list[0], rho_list[1], nsamples_from_rho, endpoint=True))
    
    sampled_alphas = list(np.geomspace(alpha_list[0], alpha_list[1], nsamples_from_alpha, endpoint=True))

    min_crit = 10000
    min_crit_embedding = 0
    
    for alpha in list(reversed(sampled_alphas)):
    
        for rho in list(reversed(sampled_rhos)):

            output = approximate_PageRank(g,ref_node,cpp=cpp,method=localmethod,alpha=alpha,rho=rho,normalize=normalize,normalized_objective=normalized_objective,epsilon=epsilon,iterations=iterations) 

            conductance = g.compute_conductance(output[0])

            crit = conductance
            if crit <= min_crit:
                min_crit = crit
                min_crit_embedding = output
    
    output_mqi = flow_clustering(g,min_crit_embedding[0],method="mqi_weighted")
    
    return output_mqi

def find_clusters(g, 
                    nclusters, 
                    rho_list, 
                    alpha_list,
                    localmethod: str = 'l1reg-rand', 
                    normalize: bool = False, 
                    normalized_objective: bool = False, 
                    cpp: bool = True,
                    epsilon: float = 1.0e-2, 
                    iterations: int = 10000000,
                    nsamples_from_rho: int = 50,
                    nsamples_from_alpha: int = 50,
                    linkage: str = 'average',
                    norm_type: int = 2,
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
        This is an interval of rhos, the regularization parameter for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.
        
    alpha_list: 2D list of floats
        This is an interval of alphas, the teleportation parameter for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.
        The smaller the more global the personalized PageRank vector is.

    Parameters (optional)
    ---------------------
        
    nsamples_from_rho: int
        Number of samples of rho parameters to be selected from interval rho_list.
        
    nsamples_from_alpha: int
        Number of samples of alpha parameters to be selected from interval alpha_list.

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
        
    cpp: bool
        Default = True
        If true calls the cpp code for approximate pagerank, otherwise, it calls the python code.

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
        
    norm_type: int
        Default = 2
        Norm for normalization of the embeddings.
        
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
        results = Parallel(n_jobs=njobs, prefer=prefer, backend=backend)(delayed(compute_embedding)(g,node,rho_list,alpha_list,nsamples_from_rho,nsamples_from_alpha,localmethod,normalize,normalized_objective,epsilon,iterations,cpp) for node in range(n))
    else:
        results =[compute_embedding(g,node,rho_list,alpha_list,nsamples_from_rho,nsamples_from_alpha,localmethod,alpha,normalize,normalized_objective,epsilon,iterations,cpp) for node in range(n)]
        
    sum_ = 0
    JA = [0]
    IA = []
    A  = []
    for data in results:
        vec = data[1]/np.linalg.norm(data[1],norm_type)
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

def compute_all_embeddings(g, 
                    rho_list, 
                    alpha_list,
                    localmethod: str = 'l1reg-rand', 
                    normalize: bool = False, 
                    normalized_objective: bool = False, 
                    cpp: bool = True,
                    epsilon: float = 1.0e-2, 
                    iterations: int = 10000000,
                    nsamples_from_rho: int = 50,
                    nsamples_from_alpha: int = 50,
                    njobs: int = 1, 
                    prefer: str = 'threads', 
                    backend: str = 'multiprocessing'):
    """
    This method runs local graph clustering for each node in the graph in parallel.
    Returns the embeddings for each node in a list. Each element of the list corresponds to an embedding
    of a node.

    Parameters
    ----------

    g: GraphLocal

    rho_list: 2D list of floats
        This is an interval of rhos, the regularization parameter for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.
        
    alpha_list: 2D list of floats
        This is an interval of alphas, the teleportation parameter for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.
        The smaller the more global the personalized PageRank vector is.

    Parameters (optional)
    ---------------------
        
    nsamples_from_rho: int
        Number of samples of rho parameters to be selected from interval rho_list.
        
    nsamples_from_alpha: int
        Number of samples of alpha parameters to be selected from interval alpha_list.

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
        
    cpp: bool
        Default = True
        If true calls the cpp code for approximate pagerank, otherwise, it calls the python code.
        
    njobs: int
        Default = 1
        Number of jobs to be run in parallel
        
    prefer, backend: str
        Check documentation of https://joblib.readthedocs.io/en/latest/

    Returns
    -------

    embeddings: list of arrays
        Each element corresponds to an embedding of a node. 
    """
    
    n = g._num_vertices
    
#     is_weighted = g._weighted
    
    if njobs > 1:
        embeddings = Parallel(n_jobs=njobs, prefer=prefer, backend=backend)(delayed(compute_embedding)(g,node,rho_list,alpha_list,nsamples_from_rho,nsamples_from_alpha,localmethod,normalize,normalized_objective,epsilon,iterations,cpp) for node in range(n))
    else:
        embeddings =[compute_embedding(g,node,rho_list,alpha_list,nsamples_from_rho,nsamples_from_alpha,localmethod,normalize,normalized_objective,epsilon,iterations,cpp) for node in range(n)]
    
    return embeddings

def normalize_embeddings(g, embeddings, 
                    norm_type: int = 2):
    """
    Normalize the embeddings.

    Parameters
    ----------

    g: GraphLocal

    embeddings: list of arrays
        Each element corresponds to an embedding of a node.
        

    Parameters (optional)
    ---------------------
        
    norm_type: int
        Default = 2
        Norm for normalization of the embeddings.

    Returns
    -------

    X: csc matrix
    The embeddings matrix. Each row corresponds to an embedding of a node. 
    
    """
    n = g._num_vertices
    
    sum_ = 0
    JA = [0]
    IA = []
    A  = []
    for data in embeddings:
        vec = data[1]/np.linalg.norm(data[1],norm_type)
        how_many = len(data[0])
        sum_ += how_many
        JA.append(sum_)
        IA.extend(list(data[0]))
        A.extend(list(vec))
    
    X = sp.sparse.csc_matrix((A, IA, JA), shape=(n, n))
    
    X = X.transpose().tocsr()
    
#     Z = pairwise_distances(X, metric=metric, n_jobs=njobs)
    
    return X

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
                    alpha_list,
                    localmethod: str = 'l1reg-rand', 
                    normalize: bool = False, 
                    normalized_objective: bool = False, 
                    cpp: bool = True,
                    epsilon: float = 1.0e-2, 
                    iterations: int = 10000000,
                    nsamples_from_rho: int = 50,
                    nsamples_from_alpha: int = 50,
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
        This is an interval of rhos, the regularization parameter for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.
        
    alpha_list: 2D list of floats
        This is an interval of alphas, the teleportation parameter for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.
        The smaller the more global the personalized PageRank vector is.

    Parameters (optional)
    ---------------------
        
    nsamples_from_rho: int
        Number of samples of rho parameters to be selected from interval rho_list.
        
    nsamples_from_alpha: int
        Number of samples of alpha parameters to be selected from interval alpha_list.

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
        
    cpp: bool
        Default = True
        If true calls the cpp code for approximate pagerank, otherwise, it calls the python code.
        
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
    
    g_copy = GraphLocal.from_sparse_adjacency(g.adjacency_matrix)
    candidates = list(range(g_copy._num_vertices))

    labels = np.zeros(g_copy._num_vertices,dtype=np.int32)
    info = []
    ct = 0

    while True:
        
        if njobs > 1:
            select_from = list(range(g_copy._num_vertices))
            ref_nodes = random.sample(select_from, min(how_many_in_parallel,len(select_from)))
            
            results = Parallel(n_jobs=njobs, prefer=prefer, backend=backend)(delayed(compute_embedding)(g_copy,node,rho_list,alpha_list,nsamples_from_rho,nsamples_from_alpha,localmethod,normalize,normalized_objective,epsilon,iterations,cpp) for node in ref_nodes)
        else:
            select_from = list(range(g_copy._num_vertices))
            ref_nodes = random.sample(select_from, njobs)
            
            results =[compute_embedding(g_copy,node,rho_list,alpha_list,nsamples_from_rho,nsamples_from_alpha,localmethod,normalize,normalized_objective,epsilon,iterations,cpp) for node in ref_nodes]
    
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

        g_copy = GraphLocal.from_sparse_adjacency(A)
        
        print ("Percentage completed: ", 100-len(candidates)/g._num_vertices*100, end="\r")
        
        if len(candidates) <= g._num_vertices*ratio:
            for i in candidates:
                labels[i] = ct
                ct += 1
            return labels, info
        
        
def graph_segmentation_with_improve(g, 
                    rho_list, 
                    alpha_list,
                    localmethod: str = 'l1reg-rand', 
                    normalize: bool = False, 
                    normalized_objective: bool = False, 
                    cpp: bool = True,
                    epsilon: float = 1.0e-2, 
                    iterations: int = 10000000,
                    nsamples_from_rho: int = 50,
                    nsamples_from_alpha: int = 50,
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
        This is an interval of rhos, the regularization parameter for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.
        
    alpha_list: 2D list of floats
        This is an interval of alphas, the teleportation parameter for l1-regularized PageRank.
        The first element should be smaller than the second elelement of the list.
        The smaller the more global the personalized PageRank vector is.

    Parameters (optional)
    ---------------------
        
    nsamples_from_rho: int
        Number of samples of rho parameters to be selected from interval rho_list.
        
    nsamples_from_alpha: int
        Number of samples of alpha parameters to be selected from interval alpha_list.

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
        
    cpp: bool
        Default = True
        If true calls the cpp code for approximate pagerank, otherwise, it calls the python code.
        
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

    g_copy = GraphLocal.from_sparse_adjacency(g.adjacency_matrix)
    candidates = list(range(g_copy._num_vertices))

    labels = np.zeros(g_copy._num_vertices,dtype=np.int32)
    info = []
    ct = 0

    while True:
        
        if njobs > 1:
            select_from = list(range(g_copy._num_vertices))
            ref_nodes = random.sample(select_from, min(how_many_in_parallel,len(select_from)))
            
            results = Parallel(n_jobs=njobs, prefer=prefer, backend=backend)(delayed(compute_embedding_and_improve)(g_copy,node,rho_list,alpha_list,nsamples_from_rho,nsamples_from_alpha,localmethod,normalize,normalized_objective,epsilon,iterations,cpp) for node in ref_nodes)
        else:
            select_from = list(range(g_copy._num_vertices))
            ref_nodes = random.sample(select_from, njobs)
            
            results =[compute_embedding_and_improve(g_copy,node,rho_list,alpha_list,nsamples_from_rho,nsamples_from_alpha,localmethod,normalize,normalized_objective,epsilon,iterations,cpp) for node in ref_nodes]
    
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

        g_copy = GraphLocal.from_sparse_adjacency(A)
        
        print ("Percentage completed: ", 100-len(candidates)/g._num_vertices*100, end="\r")
        
        if len(candidates) <= g._num_vertices*ratio:
            for i in candidates:
                labels[i] = ct
                ct += 1
            return labels, info
        
def compute_embeddings_and_distances_from_region_adjacency(g,info, metric='euclidean', norm_type = 2, n_jobs=1):
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
        
        
    norm_type: int
        Default = 2
        Norm for normalization of the embeddings.
        
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
        vec = data[1]/np.linalg.norm(data[1],norm_type)
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


def semisupervised_learning_with_improve(g,truth,kwargs_list,nprocs=1):
    input_size_all = []
    l1reg_PR_all = []
    l1reg_RC_all = []
    l1reg_F1_all = []
    mqi_PR_all = []
    mqi_RC_all = []
    mqi_F1_all = []
    flow_PR_all = []
    flow_RC_all = []
    flow_F1_all = []
    def wrapper(q_in,q_out):
        while True:
            kwargs = q_in.get()
            if kwargs is None:
                break
            delta = kwargs["delta"]
            del kwargs["delta"]
            ntrials = 0
            input_size_curr = []
            l1reg_PR_curr = []
            l1reg_RC_curr = []
            l1reg_F1_curr = []
            mqi_PR_curr = []
            mqi_RC_curr = []
            mqi_F1_curr = []
            flow_PR_curr = []
            flow_RC_curr = []
            flow_F1_curr = []
            while ntrials < 20:
                seed_node = np.random.choice(truth)
                l1reg_output = spectral_clustering(g,[seed_node],**kwargs)[0]
                if len(l1reg_output) == 0:
                    continue
                input_size_curr.append(len(l1reg_output))
                if g._weighted:
                    mqi_output = flow_clustering(g,l1reg_output,method="mqi_weighted")[0]
                    flow_output = flow_clustering(g,l1reg_output,method="flow_weighted",delta=delta)[0]
                else:
                    mqi_output = flow_clustering(g,l1reg_output,method="mqi")[0]
                    flow_output = flow_clustering(g,l1reg_output,method="flow",delta=delta)[0]
                l1reg_PR = len(set(truth).intersection(l1reg_output))/(1.0*len(l1reg_output))
                l1reg_RC = len(set(truth).intersection(l1reg_output))/(1.0*len(truth))
                l1reg_PR_curr.append(l1reg_PR)
                l1reg_RC_curr.append(l1reg_RC)
                l1reg_F1_curr.append(2*(l1reg_PR*l1reg_RC)/(l1reg_PR+l1reg_RC)) if (l1reg_PR+l1reg_RC) > 0 else 0
                mqi_PR = len(set(truth).intersection(mqi_output))/(1.0*len(mqi_output))
                mqi_RC = len(set(truth).intersection(mqi_output))/(1.0*len(truth))
                mqi_PR_curr.append(mqi_PR)
                mqi_RC_curr.append(mqi_RC)
                mqi_F1_curr.append(2*(mqi_PR*mqi_RC)/(mqi_PR+mqi_RC)) if (mqi_PR+mqi_RC) > 0 else 0
                flow_PR = len(set(truth).intersection(flow_output))/(1.0*len(flow_output))
                flow_RC = len(set(truth).intersection(flow_output))/(1.0*len(truth))
                flow_PR_curr.append(flow_PR)
                flow_RC_curr.append(flow_RC)
                flow_F1_curr.append(2*(flow_PR*flow_RC)/(flow_PR+flow_RC)) if (flow_PR+flow_RC) > 0 else 0
                ntrials += 1
            q_out.put((np.mean(input_size_curr),np.std(input_size_curr),
                       np.mean(l1reg_PR_curr),np.std(l1reg_PR_curr),
                       np.mean(l1reg_RC_curr),np.std(l1reg_RC_curr),
                       np.mean(l1reg_F1_curr),np.std(l1reg_F1_curr),
                       np.mean(mqi_PR_curr),np.std(mqi_PR_curr),
                       np.mean(mqi_RC_curr),np.std(mqi_RC_curr),
                       np.mean(mqi_F1_curr),np.std(mqi_F1_curr),
                       np.mean(flow_PR_curr),np.std(flow_PR_curr),
                       np.mean(flow_RC_curr),np.std(flow_RC_curr),
                       np.mean(flow_F1_curr),np.std(flow_F1_curr)))
    q_in,q_out = mp.Queue(),mp.Queue()
    for kwargs in kwargs_list:
        q_in.put(kwargs)
    for _ in range(nprocs):
        q_in.put(None)
    procs = [mp.Process(target=wrapper,args=(q_in,q_out)) for _ in range(nprocs)]
    for p in procs:
        p.start()
    ncounts = 0
    while ncounts < len(kwargs_list):
        output = q_out.get()
        input_size_all.append((output[0],output[1]))
        l1reg_PR_all.append((output[2],output[3]))
        l1reg_RC_all.append((output[4],output[5]))
        l1reg_F1_all.append((output[6],output[7]))
        mqi_PR_all.append((output[8],output[9]))
        mqi_RC_all.append((output[10],output[11]))
        mqi_F1_all.append((output[12],output[13]))
        flow_PR_all.append((output[14],output[15]))
        flow_RC_all.append((output[16],output[17]))
        flow_F1_all.append((output[18],output[19]))
        ncounts += 1
    for p in procs:
        p.join()
    return locals()


def semisupervised_learning(g,truth_dict,kwargs_list,nprocs=1,size_ratio=0.1,use_bfs=True,flowmethod="mqi_weighted",use_spectral=True):
    l1reg_PR_all = np.zeros((len(kwargs_list),3))
    l1reg_RC_all = np.zeros((len(kwargs_list),3))
    l1reg_F1_all = np.zeros((len(kwargs_list),3))
    flow_PR_all = np.zeros((len(kwargs_list),3))
    flow_RC_all = np.zeros((len(kwargs_list),3))
    flow_F1_all = np.zeros((len(kwargs_list),3))
    flow_PR_all1 = np.zeros((len(kwargs_list),3))
    flow_RC_all1 = np.zeros((len(kwargs_list),3))
    flow_F1_all1 = np.zeros((len(kwargs_list),3))
    l1reg_PR_curr = defaultdict(list)
    l1reg_RC_curr = defaultdict(list)
    l1reg_F1_curr = defaultdict(list)
    flow_PR_curr = defaultdict(list)
    flow_RC_curr = defaultdict(list)
    flow_F1_curr = defaultdict(list)
    flow_PR_curr1 = defaultdict(list)
    flow_RC_curr1 = defaultdict(list)
    flow_F1_curr1 = defaultdict(list)
    total_vol = np.sum(g.d)
    def wrapper(pid,q_in,q_out):
        while True:
            kwargs,kwargs_id,trial_id,delta,delta1,ratio = q_in.get()
            if kwargs is None:
                break
            nlabels = len(list(truth_dict.keys()))
            l1reg_labels = np.zeros(g._num_vertices) - 1
            true_labels = np.zeros(g._num_vertices) - 1
            flow_labels = np.zeros(g._num_vertices) - 1
            flow_labels1 = np.zeros(g._num_vertices) - 1
            ranking = np.zeros(g._num_vertices) - 1
            npositives = 0
            for lid,label in enumerate(sorted(list(truth_dict.keys()))):
                truth = truth_dict[label]
                npositives += len(truth)
                true_labels[truth] = lid
                nseeds = int(ratio*len(truth))
                np.random.seed(1000*kwargs_id+10*trial_id+lid)
                seeds = np.random.choice(truth,nseeds)
                if use_spectral:
                    l1reg_ids,l1reg_vals = approximate_PageRank(g,seeds,**kwargs)
                    sorted_indices = np.argsort(-1*l1reg_vals)
                    for i,idx in enumerate(sorted_indices):
                        if ranking[l1reg_ids[idx]] == -1 or i < ranking[l1reg_ids[idx]]:
                            ranking[l1reg_ids[idx]] = i
                            l1reg_labels[l1reg_ids[idx]] = lid
                #flow_output1 = flow_clustering(g,seeds,method=flowmethod,delta=curr_vol/(total_vol-curr_vol))[0]
                if use_bfs:
                    seeds = seed_grow_bfs_steps(g,seeds,1)
                flow_output = flow_clustering(g,seeds,method=flowmethod,delta=delta)[0]
                flow_output1 = flow_clustering(g,seeds,method=flowmethod,delta=delta1)[0]
                curr_vol = np.sum(g.d[seeds])
                for i,idx in enumerate(flow_output):
                    if flow_labels[idx] == -1:
                        flow_labels[idx] = lid
                    else:
                        flow_labels[idx] = nlabels + 1
                for i,idx in enumerate(flow_output1):
                    if flow_labels1[idx] == -1:
                        flow_labels1[idx] = lid
                    else:
                        flow_labels1[idx] = nlabels + 1
            if use_spectral:
                l1reg_PR = np.sum((l1reg_labels == true_labels))/(1.0*np.sum(l1reg_labels!=-1))
                l1reg_RC = np.sum((l1reg_labels == true_labels))/(1.0*npositives)
                l1reg_F1 = 2*(l1reg_PR*l1reg_RC)/(l1reg_PR+l1reg_RC) if (l1reg_PR+l1reg_RC) > 0 else 0
            else:
                l1reg_PR,l1reg_RC,l1reg_F1 = 0,0,0
            # l1reg_PR_curr.append(l1reg_PR)
            # l1reg_RC_curr.append(l1reg_RC)
            # l1reg_F1_curr.append() 
            flow_PR = np.sum((flow_labels == true_labels))/(1.0*np.sum(flow_labels!=-1))
            flow_RC = np.sum((flow_labels == true_labels))/(1.0*npositives)
            flow_F1 = 2*(flow_PR*flow_RC)/(flow_PR+flow_RC) if (flow_PR+flow_RC) > 0 else 0
            flow_PR1 = np.sum((flow_labels1 == true_labels))/(1.0*np.sum(flow_labels1!=-1))
            flow_RC1 = np.sum((flow_labels1 == true_labels))/(1.0*npositives)
            flow_F11 = 2*(flow_PR1*flow_RC1)/(flow_PR1+flow_RC1) if (flow_PR1+flow_RC1) > 0 else 0
            # flow_PR_curr.append(flow_PR)
            # flow_RC_curr.append(flow_RC)
            # flow_F1_curr.append() 
            q_out.put((kwargs_id,trial_id,l1reg_PR,l1reg_RC,l1reg_F1,flow_PR,flow_RC,flow_F1,flow_PR1,flow_RC1,flow_F11))
    q_in,q_out = mp.Queue(),mp.Queue()
    ntrials = 30
    for kwargs_id in range(len(kwargs_list)):
        kwargs = copy.deepcopy(kwargs_list[kwargs_id])
        delta = kwargs["delta"]
        del kwargs["delta"]
        delta1 = kwargs["delta1"]
        del kwargs["delta1"]
        ratio = kwargs["ratio"]
        del kwargs["ratio"]
        for trial_id in range(ntrials):
            q_in.put((kwargs,kwargs_id,trial_id,delta,delta1,ratio))
    for _ in range(nprocs):
        q_in.put((None,None,None,None,None,None))
    procs = [mp.Process(target=wrapper,args=(pid,q_in,q_out)) for pid in range(nprocs)]
    for p in procs:
        p.start()
    ncounts = 0
    while ncounts < len(kwargs_list)*ntrials:
        if ncounts%10 == 0:
            print("Finished "+str(ncounts)+"/"+str(len(kwargs_list)*ntrials)+" experiments.")
        kwargs_id,trial_id,l1reg_PR,l1reg_RC,l1reg_F1,flow_PR,flow_RC,flow_F1,flow_PR1,flow_RC1,flow_F11 = q_out.get()
        l1reg_PR_curr[kwargs_id].append(l1reg_PR)
        l1reg_RC_curr[kwargs_id].append(l1reg_RC)
        l1reg_F1_curr[kwargs_id].append(l1reg_F1)
        flow_PR_curr[kwargs_id].append(flow_PR)
        flow_RC_curr[kwargs_id].append(flow_RC)
        flow_F1_curr[kwargs_id].append(flow_F1)
        flow_PR_curr1[kwargs_id].append(flow_PR1)
        flow_RC_curr1[kwargs_id].append(flow_RC1)
        flow_F1_curr1[kwargs_id].append(flow_F11)
        if trial_id == ntrials - 1:
            l1reg_PR_all[kwargs_id] = [np.median(l1reg_PR_curr[kwargs_id]),np.percentile(l1reg_PR_curr[kwargs_id],q=20),
                np.percentile(l1reg_PR_curr[kwargs_id],q=80)]
            l1reg_RC_all[kwargs_id] = [np.median(l1reg_RC_curr[kwargs_id]),np.percentile(l1reg_RC_curr[kwargs_id],q=20),
                np.percentile(l1reg_RC_curr[kwargs_id],q=80)]
            l1reg_F1_all[kwargs_id] = [np.median(l1reg_F1_curr[kwargs_id]),np.percentile(l1reg_F1_curr[kwargs_id],q=20),
                np.percentile(l1reg_F1_curr[kwargs_id],q=80)]
            flow_PR_all[kwargs_id] = [np.median(flow_PR_curr[kwargs_id]),np.percentile(flow_PR_curr[kwargs_id],q=20),
                np.percentile(flow_PR_curr[kwargs_id],q=80)]
            flow_RC_all[kwargs_id] = [np.median(flow_RC_curr[kwargs_id]),np.percentile(flow_RC_curr[kwargs_id],q=20),
                np.percentile(flow_RC_curr[kwargs_id],q=80)]
            flow_F1_all[kwargs_id] = [np.median(flow_F1_curr[kwargs_id]),np.percentile(flow_F1_curr[kwargs_id],q=20),
                np.percentile(flow_F1_curr[kwargs_id],q=80)]
            flow_PR_all1[kwargs_id] = [np.median(flow_PR_curr1[kwargs_id]),np.percentile(flow_PR_curr1[kwargs_id],q=20),
                np.percentile(flow_PR_curr1[kwargs_id],q=80)]
            flow_RC_all1[kwargs_id] = [np.median(flow_RC_curr1[kwargs_id]),np.percentile(flow_RC_curr1[kwargs_id],q=20),
                np.percentile(flow_RC_curr1[kwargs_id],q=80)]
            flow_F1_all1[kwargs_id] = [np.median(flow_F1_curr1[kwargs_id]),np.percentile(flow_F1_curr1[kwargs_id],q=20),
                np.percentile(flow_F1_curr1[kwargs_id],q=80)]
        ncounts += 1
    for p in procs:
        p.join()
    
    del procs
    del p
    del q_in
    del q_out
    del wrapper
    return locals()

def seed_grow_bfs_steps(g,seeds,steps):
    """
    grow the initial seed set through BFS until its size reaches 
    a given ratio of the total number of nodes.
    """
    Q = queue.Queue()
    visited = np.zeros(g._num_vertices)
    visited[seeds] = 1
    for s in seeds:
        Q.put(s)
    if isinstance(seeds,np.ndarray):
        seeds = seeds.tolist()
    else:
        seeds = list(seeds)
    for step in range(steps):
        for k in range(Q.qsize()):
            node = Q.get()
            si,ei = g.adjacency_matrix.indptr[node],g.adjacency_matrix.indptr[node+1]
            neighs = g.adjacency_matrix.indices[si:ei]
            for i in range(len(neighs)):
                if visited[neighs[i]] == 0:
                    visited[neighs[i]] = 1
                    seeds.append(neighs[i])
                    Q.put(neighs[i])
    return seeds

def seed_grow_bfs_size(g,seeds,nseeds):
    """
    grow the initial seed set through BFS until its size reaches 
    a given ratio of the total number of nodes.
    """
    Q = queue.Queue()
    visited = np.zeros(g._num_vertices)
    visited[seeds] = 1
    for s in seeds:
        Q.put(s)
    if isinstance(seeds,np.ndarray):
        seeds = seeds.tolist()
    else:
        seeds = list(seeds)
    while len(seeds) < nseeds:
        node = Q.get()
        si,ei = g.adjacency_matrix.indptr[node],g.adjacency_matrix.indptr[node+1]
        neighs = g.adjacency_matrix.indices[si:ei]
        for i in range(len(neighs)):
            if visited[neighs[i]] == 0:
                visited[neighs[i]] = 1
                seeds.append(neighs[i])
                Q.put(neighs[i])
            if len(seeds) == nseeds:
                break
    return seeds