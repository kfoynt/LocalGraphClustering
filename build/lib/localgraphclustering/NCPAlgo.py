import numpy as np
import pandas as pd
import time
from collections import namedtuple
import threading
import warnings
import math

from localgraphclustering import capacity_releasing_diffusion_fast
from localgraphclustering import MQI_fast
from localgraphclustering import l1_regularized_PageRank_fast
from localgraphclustering import sweepCut_fast
from localgraphclustering import approximate_PageRank_Clustering

def graph_set_scores(graph, R):
    voltrue = sum(graph.d[R])     
    v_ones_R = np.zeros(graph._num_vertices)
    v_ones_R[R] = 1
    cut = voltrue - np.dot(v_ones_R,graph.adjacency_matrix.dot(v_ones_R.T))

    voleff = min(voltrue,graph.vol_G - voltrue)
    
    sizetrue = len(R)
    sizeeff = sizetrue
    if voleff < voltrue:
        sizeeff = graph._num_vertices - sizetrue
        
    # remove the stuff we don't want returned...
    del R
    del graph
    del v_ones_R

    edgestrue = voltrue - cut
    edgeseff = voleff - cut
    
    cond = cut / voleff if voleff != 0 else 1
    isop = cut / sizeeff
    
    # make a dictionary out of local variables
    return locals()

def crd_wrapper(G,R,U=3,h=10,w=2,iterations=20):
    crd_fast = capacity_releasing_diffusion_fast.Capacity_Releasing_Diffusion_fast()
    #crd = localgraphclustering.capacity_releasing_diffusion_fast()
    return [list(crd_fast.produce([G],R,U,h,w,iterations)[0])]

def mqi_wrapper(G,R):
    MQI_fast_obj = MQI_fast.MQI_fast()
    output_MQI_fast = MQI_fast_obj.produce([G],[R])
    #print(output_MQI_fast)
    return [output_MQI_fast[0][0].tolist()]

def l1reg_wrapper(G,R,epsilon=1.0e-1,iterations=1000):
    l1reg_fast = l1_regularized_PageRank_fast.L1_regularized_PageRank_fast()
    sc_fast = sweepCut_fast.SweepCut_fast()
    S = []
    rho_list = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4]
    for rho in rho_list:
        a_list = [1-0.99]
        for alpha in a_list:
            output_l1reg_fast = l1reg_fast.produce([G],R,alpha=alpha,rho=rho,epsilon=epsilon,iterations=iterations)
            output_sc_fast = sc_fast.produce([G],p=output_l1reg_fast[0])
            S.append(output_sc_fast[0][0])
    return S

def approxPageRank_wrapper(G,R,iterations=10000):
    pr_clustering = approximate_PageRank_Clustering.Approximate_PageRank_Clustering()
    S = []
    rho_list = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4]
    for rho in rho_list:
        a_list = [1-0.99]
        for alpha in a_list:
            output_pr_clustering = pr_clustering.produce([G],R,alpha=alpha,rho=rho,iterations=iterations)
            S.append(output_pr_clustering[0])
    return S

"""
def acl_wrapper(G,R,eps):
    acl_fast = localclutering.approximate_PageRank_fast.Approximate_PageRank_fast()
    return list(acl_fast([G],[R],alpha=0.01,rho=eps,))
"""    

def ncp_experiment(ncpdata,R,func,method_stats):
    if ncpdata.input_stats:
        input_stats = graph_set_scores(ncpdata.graph, R)
        for F in ncpdata.set_funcs: # build the list of keys for set_funcs
            input_stats.update(F(ncpdata.graph, R))
        input_stats = {"input_" + str(key):value for key,value in input_stats.items() } # add input prefix
    else:
        input_stats = {}
        
    start = time.time()
    Sets = func(ncpdata.graph, R)
    ret_stats = []
    dt = time.time() - start
    for S in Sets:
        if len(S) == 0:
            continue
        output_stats = graph_set_scores(ncpdata.graph, S)
        for F in ncpdata.set_funcs: # build the list of keys for set_funcs
            output_stats.update(F(ncpdata.graph, S))
        output_stats = {"output_" + str(key):value for key,value in output_stats.items() } # add output prefix

        method_stats['methodfunc']  = func
        method_stats['time'] = dt
        ret_stats.append(ncpdata.record(**input_stats, **output_stats, **method_stats))
    return ret_stats

def ncp_node_worker(ncpdata,sets,func,timeout_ncp):
    start = time.time()
    setno = 0
    for R in sets:
        #print("setno = %i"%(setno))
        setno += 1
        
        method_stats = {'input_set_type': 'node', 'input_set_params':R[0]}
        ncpdata.results.extend(ncp_experiment(ncpdata, R, func, method_stats))
        
        end = time.time()
        if end - start > timeout_ncp:
            break

# todo avoid code duplication 
def ncp_neighborhood_worker(ncpdata,sets,func,timeout_ncp):
    start = time.time()
    setno = 0
    for R in sets:
        #print("setno = %i"%(setno))
        setno += 1
        
        R = R.copy() # duplicate so we don't keep extra weird data around
        node = R[0]
        R.extend(ncpdata.graph.neighbors(R[0]))
        method_stats = {'input_set_type': 'neighborhood', 'input_set_params':node}
        
        ncpdata.results.extend(ncp_experiment(ncpdata, R, func, method_stats))
        
        end = time.time()
        if end - start > timeout_ncp:
            break          
            
# todo avoid code duplication 
def ncp_set_worker(ncpdata,setnos,func,timeout_ncp):
    start = time.time()
    setno = 0
    for id in setnos:
        #print("setno = %i"%(setno))
        setno += 1
        R = ncpdata.sets[id]
        R = R.copy() # duplicate so we don't keep extra weird data around
        R.extend(ncpdata.graph.neighbors(R[0]))
        method_stats = {'input_set_type': 'set', 'input_set_params':id}
        
        ncpdata.results.extend(ncp_experiment(ncpdata, R, func, method_stats))
        
        end = time.time()
        if end - start > timeout_ncp:
            break 