import numpy as np
import networkx as nx

from localgraphclustering import graph_class_local
from localgraphclustering import l1_regularized_PageRank_fast
from localgraphclustering import sweepCut_fast
from localgraphclustering.plot_ncp import plot_ncp_vol
from localgraphclustering.plot_ncp import plot_ncp_size

import time
import threading
import math

conductance_vs_vol = []
conductance_vs_size = []
fista_counter = 0
lock = threading.RLock()
start = 0
g_copy = 0
l1reg_fast = 0
sc_fast = 0
cmp = 0

def worker(nodes,timeout_ncp,epsilon,iterations,timeout):
    global fista_counter
    fista_counter = 0
    for node in nodes:

    
        rho_list = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4]
        
        for rho in rho_list:
                
            #a_list = np.arange(1.0e-2,0.999,0.09)
            a_list = [1-0.99]

            for alpha in a_list:
                        
                lock.acquire()
                try:    
                    fista_counter = fista_counter + 1
                    #print(fista_counter)
                finally:
                    lock.release()


                output_l1reg_fast = l1reg_fast.produce([g_copy],[node],alpha=alpha,rho=rho,epsilon=epsilon,iterations=iterations,timeout=timeout)
                    
                output_sc_fast = sc_fast.produce([g_copy],p=output_l1reg_fast[0])

                S_l1pr = output_sc_fast[0][0]
                cond = output_sc_fast[0][1]

                vol = sum(g_copy.d[S_l1pr])
                size = len(S_l1pr)

                if vol in conductance_vs_vol[cmp]:
                    if cond <= conductance_vs_vol[cmp][vol]:
                        conductance_vs_vol[cmp][vol] = cond
                else:
                    conductance_vs_vol[cmp][vol] = cond  

                if size in conductance_vs_size[cmp]:
                    if cond <= conductance_vs_size[cmp][size]:
                        conductance_vs_size[cmp][size] = cond
                else:
                    conductance_vs_size[cmp][size] = cond     

                end = time.time()
                        
        if end - start > timeout_ncp:
            break


def ncp_algo(g, ratio, timeout = 10, timeout_ncp = 1000, iterations = 1000, epsilon = 1.0e-1, nthreads = 20, multi_threads = True):
    """
    Network Community Profile for all connected components of the graph. For details please refer to: 
    Jure Leskovec, Kevin J Lang, Anirban Dasgupta, Michael W Mahoney. Community structure in 
    large networks: Natural cluster sizes and the absence of large well-defined clusters.
    The NCP is computed for each connected component of the given graph.

    Parameters
    ----------  

    ratio: float
        Ratio of nodes to be used for computation of NCP.
        It should be between 0 and 1.

    Parameters (optional)
    ---------------------

    epsilon: float
        default = 1.0e-1
        Termination tolerance for l1-regularized PageRank solver.

    iterations: int
        default = 10000
        Maximum number of iterations of l1-regularized PageRank solver.

    timeout_ncp: float
        default = 1000
        Maximum time in seconds for NCP calculation.

    timeout: float
        default = 10
        Maximum time in seconds for each algorithm run during the NCP calculation.

    Returns
    -------

    conductance_vs_vol: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. It can be used to plot the conductance vs volume NCP.

    conductance_vs_size: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are conductance. It can be used to plot the conductance vs volume NCP.
    """
    global start, g_copy, l1reg_fast, sc_fast, cmp         
    if ratio < 0 or ratio > 1:
        print("Ratio must be between 0 and 1.")
        return []        
    
    l1reg_fast = l1_regularized_PageRank_fast.L1_regularized_PageRank_fast()
    sc_fast = sweepCut_fast.SweepCut_fast()
        
    g.connected_components()
        
    number_of_components = g.number_of_components
        
    if number_of_components <= 0:
        print("There are no connected components in the given graph")
        return
        
    for i in range(number_of_components):
        conductance_vs_vol.append({})
        conductance_vs_size.append({})
        
    start = time.time()    
        
    for cmp in range(number_of_components):
            
        nodes_of_component = list(g.components[cmp])
        g_copy = graph_class_local.GraphLocal()
        g_copy.adjacency_matrix = g.adjacency_matrix[nodes_of_component,:].tocsc()[:,nodes_of_component].tocsr()
        g_copy.compute_statistics()          
        
        n = g_copy.adjacency_matrix.shape[0]

        n_nodes = min(np.ceil(ratio*n),n)
        n_nodes = int(n_nodes)
            
        #p = g_copy.d/g_copy.vol_G
        #nodes = np.random.choice(np.arange(0,n), size=n_nodes, replace=False, p=p)
        nodes = np.random.choice(np.arange(0,n), size=n_nodes, replace=False)
        
        threads = []
        #print(n_nodes)
        #print(for_each_worker)
        
        if (not multi_threads):
            nthreads = 1
        for_each_worker = math.floor(n_nodes/nthreads)
        for i in range(nthreads):
            start_pos = for_each_worker*i
            end_pos = min(for_each_worker*(i+1),n_nodes)
            t = threading.Thread(target=worker,args=(nodes[start_pos:end_pos],timeout_ncp,
                epsilon,iterations,timeout))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        print("# of calls to FISTA:", fista_counter)
        
        print('NCP plots for component: ' + str(cmp))
        plot_ncp_vol(conductance_vs_vol[cmp])
        plot_ncp_size(conductance_vs_size[cmp])

    return (conductance_vs_vol,conductance_vs_size)
