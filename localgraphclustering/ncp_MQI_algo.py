import numpy as np
import networkx as nx

from localgraphclustering import graph_class_local
from localgraphclustering import MQI_fast
from localgraphclustering.plot_ncp import plot_ncp_MQI_vol
from localgraphclustering.plot_ncp import plot_ncp_MQI_size
from localgraphclustering.plot_ncp import plot_ncp_conductance_node
from localgraphclustering.plot_ncp import plot_ncp_isoperimetry_node

import time
import threading
import math

conductance_vs_vol_R = []
isoperimetry_vs_size_R = []
conductance_vs_vol_MQI = []
isoperimetry_vs_size_MQI = []
conductance_vs_node_R = []
isoperimetry_vs_node_R = []
conductance_vs_node_MQI = []
isoperimetry_vs_node_MQI = []
start = 0
g_copy = 0
g_complete = 0
MQI_fast_obj = 0
n = 0
cmp = 0

def worker(nodes,timeout_ncp):
    for node in nodes:
        R = g_copy.adjacency_matrix[:,node].nonzero()[0].tolist()
        R.extend([node])
        output_MQI_fast = MQI_fast_obj.produce([g_complete],[R])
        output = output_MQI_fast[0][0].tolist()

        v_ones_R = np.zeros(n)
        v_ones_R[R] = 1

        v_ones_MQI = np.zeros(n)
        v_ones_MQI[output] = 1

        vol_R = sum(g_copy.d[R])
        size_R = len(R)

        vol_MQI = sum(g_copy.d[output])
        size_MQI = len(output)        

        cut_R = vol_R - np.dot(v_ones_R,g_copy.adjacency_matrix.dot(v_ones_R.T))
        cut_MQI = vol_MQI - np.dot(v_ones_MQI,g_copy.adjacency_matrix.dot(v_ones_MQI.T))

        cond_R = cut_R/min(vol_R,g_copy.vol_G - vol_R)
        cond_MQI = cut_MQI/min(vol_MQI,g_copy.vol_G - vol_MQI)

        conductance_vs_node_R[cmp][node] = cond_R
        conductance_vs_node_MQI[cmp][node] = cond_MQI

        isop_R = cut_R/min(size_R,n - size_R)
        isop_MQI = cut_MQI/min(size_MQI,n - size_MQI)

        isoperimetry_vs_node_R[cmp][node] = isop_R
        isoperimetry_vs_node_MQI[cmp][node] = isop_MQI

        if vol_R in conductance_vs_vol_R[cmp]:
            if cond_R <= conductance_vs_vol_R[cmp][vol_R]:
                conductance_vs_vol_R[cmp][vol_R] = cond_R
        else:
            conductance_vs_vol_R[cmp][vol_R] = cond_R  

        if size_R in isoperimetry_vs_size_R[cmp]:
            if isop_R <= isoperimetry_vs_size_R[cmp][size_R]:
                isoperimetry_vs_size_R[cmp][size_R] = isop_R
        else:
            isoperimetry_vs_size_R[cmp][size_R] = isop_R 

        if vol_R in conductance_vs_vol_MQI[cmp]:
            if cond_MQI <= conductance_vs_vol_MQI[cmp][vol_R]:
                    conductance_vs_vol_MQI[cmp][vol_R] = cond_MQI
        else:
            conductance_vs_vol_MQI[cmp][vol_R] = cond_MQI  

        if size_R in isoperimetry_vs_size_MQI[cmp]:
            if isop_MQI <= isoperimetry_vs_size_MQI[cmp][size_R]:
                isoperimetry_vs_size_MQI[cmp][size_R] = isop_MQI
        else:
            isoperimetry_vs_size_MQI[cmp][size_R] = isop_MQI
                
        end = time.time()
        if end - start > timeout_ncp:
            break


def ncp_MQI_algo(g, ratio, timeout_ncp = 1000, nthreads = 20, multi_threads = True):
    """
    Network Community Profile based on neighborhoods and MQI for all connected components of the graph.

    Parameters
    ----------  

    ratio: float
        Ratio of nodes to be used for computation of NCP.
        It should be between 0 and 1.

    Parameters (optional)
    ---------------------

    timeout_ncp: float
        default = 1000
        Maximum time in seconds for NCP calculation.

    Returns
    -------

    conductance_vs_vol_R: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. The clusters are computed using neighbors of randomly 
        selected nodes. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.

    isoperimetry_vs_size_R: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are isoperimetry. The clusters are computed using neighbors of randomly 
        selected nodes. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.
        
    conductance_vs_node_R: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. The clusters are computed using neighbors of randomly 
        selected nodes. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.

    isoperimetry_vs_node_R: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are isoperimetry. The clusters are computed using neighbors of randomly 
        selected nodes. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.
        
    conductance_vs_vol_MQI: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. The clusters are computed using neighbors of randomly 
        selected nodes on which we apply MQI. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.

    isoperimetry_vs_size_MQI: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are isoperimetry. The clusters are computed using neighbors of randomly 
        selected nodes on which we apply MQI. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.
        
    conductance_vs_node_MQI: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. The clusters are computed using neighbors of randomly 
        selected nodes on which we apply MQI. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.

    isoperimetry_vs_node_MQI: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are isoperimetry. The clusters are computed using neighbors of randomly 
        selected nodes on which we apply MQI. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.
    """
    global start, g_copy, MQI_fast_obj, cmp, g_complete, n
    g_complete = g
    if ratio < 0 or ratio > 1:
        print("Ratio must be between 0 and 1.")
        return []  

    MQI_fast_obj = MQI_fast.MQI_fast()

    g.connected_components()

    number_of_components = g.number_of_components

    for i in range(number_of_components):
        conductance_vs_vol_R.append({})
        isoperimetry_vs_size_R.append({})
        conductance_vs_vol_MQI.append({})
        isoperimetry_vs_size_MQI.append({})
        conductance_vs_node_R.append({})
        isoperimetry_vs_node_R.append({})
        conductance_vs_node_MQI.append({})
        isoperimetry_vs_node_MQI.append({})

    start = time.time()
        
    for cmp in range(number_of_components):

        nodes_of_component = list(g.components[cmp])
        g_copy = graph_class_local.GraphLocal()
        g_copy.adjacency_matrix = g.adjacency_matrix[nodes_of_component,:].tocsc()[:,nodes_of_component].tocsr()
        g_copy.compute_statistics()          

        n = g_copy.adjacency_matrix.shape[0]

        n_nodes = min(np.ceil(ratio*n),n)
        n_nodes = int(n_nodes)

        nodes = np.random.choice(np.arange(0,n), size=n_nodes, replace=False)

        threads = []

        if (not multi_threads):
            nthreads = 1
        for_each_worker = math.floor(n_nodes/nthreads)
        for i in range(nthreads):
            start_pos = for_each_worker*i
            end_pos = min(for_each_worker*(i+1),n_nodes)
            t = threading.Thread(target=worker,args=(nodes[start_pos:end_pos],timeout_ncp))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        print('NCP plots for component: ' + str(cmp))
        plot_ncp_MQI_vol(conductance_vs_vol_R[cmp],conductance_vs_vol_MQI[cmp])
        plot_ncp_MQI_size(isoperimetry_vs_size_R[cmp],isoperimetry_vs_size_MQI[cmp])
        plot_ncp_conductance_node(conductance_vs_node_R[cmp],conductance_vs_node_MQI[cmp])
        plot_ncp_isoperimetry_node(isoperimetry_vs_node_R[cmp],isoperimetry_vs_node_MQI[cmp])
        
        return (conductance_vs_vol_R,isoperimetry_vs_size_R,conductance_vs_node_R,isoperimetry_vs_node_R,conductance_vs_vol_MQI,isoperimetry_vs_size_MQI,conductance_vs_node_MQI,isoperimetry_vs_node_MQI)
