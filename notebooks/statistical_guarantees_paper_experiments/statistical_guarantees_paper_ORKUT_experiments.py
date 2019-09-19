import scipy as sp
import numpy as np
import time

import matplotlib.pyplot as plt

try:
    from localgraphclustering import *
except:
    # when the package is not installed, import the local version instead. 
    # the notebook must be placed in the original "notebooks/" folder
    sys.path.append("../")
    from localgraphclustering import * 

import time

import networkx as nx

import random

import statistics as stat_

import sys,os

data_path = os.getcwd()
name = '../datasets/com-orkut.ungraph.edgelist'
g = GraphLocal(os.path.join(data_path,name),'edgelist', "	")

comm_name = '../datasets/com-orkut.top5000.cmty.txt'
ground_truth_clusters = []
with open(comm_name, "r") as f:
    for line in f:
        new_line = []
        for i in line.split():
            if i.isdigit():
                new_line.append(int(i))
        ground_truth_clusters.append(new_line)
        
        
all_clusters = []

some_data = np.zeros((282,1))
    
counter = 0
ct = 0 
for cluster in ground_truth_clusters:
    
    if len(cluster) == 1 or len(cluster) == 0:
        counter += 1
        continue
    
#     eig, lambda_ = fiedler_local(g, cluster)
#     lambda_ = np.real(lambda_)
#     gap = lambda_/g.compute_conductance(cluster)
    cond = g.compute_conductance(cluster)
    counter += 1
    
    if cond <= 0.6 and len(cluster) >= 10:
        print("Cluster: ", counter, " conductance: ", cond, "Size: ", len(cluster))
        all_clusters.append(cluster)
        some_data[ct,0] = cond
        ct += 1
        
        
## Collect data for ACL (with rounding)

nodes = {}
external_best_cond_acl = {}
external_best_pre_cond_acl = {}
vol_best_cond_acl = {}
vol_best_pre_acl = {}
size_clust_best_cond_acl = {}
size_clust_best_pre_acl = {}
f1score_best_cond_acl = {}
f1score_best_pre_acl = {}
true_positives_best_cond_acl = {}
true_positives_best_pre_acl = {}
precision_best_cond_acl = {}
precision_best_pre_acl = {}
recall_best_cond_acl = {}
recall_best_pre_acl = {}
cuts_best_cond_acl = {}
cuts_best_pre_acl = {}
cuts_acl_ALL = {}

ct_outer = 0

number_experiments = 0

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    eigv, lambda_val = fiedler_local(g, rr)
    lambda_val = np.real(lambda_val)
    
    step = (2*lambda_val - lambda_val/2)/4
    
    a_list = np.arange(lambda_val/2,2*lambda_val,step)
    
    ct = 0
    
    start = time.time()
    
    for node in nodes[ct_outer]:
        ref_node = [node]
        
        max_precision = -1
        min_conduct = 100
        
        ct_inner = 0
        for a in a_list:
            
            if ct_outer <= 1:
                rho = 0.15/np.sum(g.d[rr])
            else:
                rho = 0.2/np.sum(g.d[rr])
            
            output_pr_clustering = approximate_PageRank(g,ref_node,method = "acl", rho=rho, alpha=a, cpp = True, normalize=True,normalized_objective=True)
            number_experiments += 1
            
            output_pr_sc = sweep_cut(g,output_pr_clustering,cpp=True)
            
            S = output_pr_sc[0]
            
            cuts_acl_ALL[ct_outer,node,ct_inner] = S
            
            size_clust_acl_ = len(S)
            
            cond_val_l1pr = g.compute_conductance(S)
            
            vol_ = sum(g.d[S])
            true_positives_acl_ = set(rr).intersection(S)
            if len(true_positives_acl_) == 0:
                true_positives_acl_ = set(ref_node)
                vol_ = g.d[ref_node][0,0]
            precision = sum(g.d[np.array(list(true_positives_acl_))])/vol_
            recall = sum(g.d[np.array(list(true_positives_acl_))])/sum(g.d[rr])
            f1_score_ = 2*(precision*recall)/(precision + recall)
            
            if f1_score_ >= max_precision:
                
                max_precision = f1_score_
                
                external_best_pre_cond_acl[ct_outer,node] = cond_val_l1pr
                vol_best_pre_acl[ct_outer,node] = vol_
                
                size_clust_best_pre_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_pre_acl[ct_outer,node] = true_positives_acl_
                precision_best_pre_acl[ct_outer,node] = precision
                recall_best_pre_acl[ct_outer,node] = recall
                f1score_best_pre_acl[ct_outer,node] = f1_score_
                
                cuts_best_pre_acl[ct_outer,node] = S
        
            if cond_val_l1pr <= min_conduct:
                
                min_conduct = cond_val_l1pr
                
                external_best_cond_acl[ct_outer,node] = cond_val_l1pr
                vol_best_cond_acl[ct_outer,node] = vol_
                
                size_clust_best_cond_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_cond_acl[ct_outer,node] = true_positives_acl_
                precision_best_cond_acl[ct_outer,node] = precision
                recall_best_cond_acl[ct_outer,node] = recall
                f1score_best_cond_acl[ct_outer,node] = f1_score_
                
                cuts_best_cond_acl[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_acl[ct_outer,node], 'f1score: ', f1score_best_cond_acl[ct_outer,node], 'precision: ', precision_best_cond_acl[ct_outer,node], 'recall: ', recall_best_cond_acl[ct_outer,node])
        ct += 1
    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time ACL with rounding: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1
    
## Performance of ACL (with rounding).

all_data = []
xlabels_ = []

print('Results for ACL with rounding')
sum_precision = 0
sum_recall = 0
sum_f1 = 0
sum_conductance = 0

info_ref_nodes = all_clusters
l_info_ref_nodes = len(info_ref_nodes)

for i in range(l_info_ref_nodes):
    temp_pre = []
    temp_rec = []
    temp_f1 = []
    temp_conductance = []
    
    for j in all_clusters[i]:
        temp_pre.append(precision_best_cond_acl[i,j])
        temp_rec.append(recall_best_cond_acl[i,j])
        temp_f1.append(f1score_best_cond_acl[i,j])
        temp_conductance.append(external_best_cond_acl[i,j])
    
    print('Feature:', i,'Precision', stat_.mean(temp_pre), 'Recall', stat_.mean(temp_rec), 'F1', stat_.mean(temp_f1), 'Cond.', stat_.mean(temp_conductance))

np.save('f1score_best_pre_acl_ORKUT', f1score_best_pre_acl) 
np.save('precision_best_pre_acl_ORKUT', precision_best_pre_acl) 
np.save('recall_best_pre_acl_ORKUT', recall_best_pre_acl) 

## Collect data for l1-reg. PR (with rounding)

nodes = {}
external_best_cond_l1reg = {}
external_best_pre_cond_l1reg = {}
vol_best_cond_l1reg = {}
vol_best_pre_l1reg = {}
size_clust_best_cond_l1reg = {}
size_clust_best_pre_l1reg = {}
f1score_best_cond_l1reg = {}
f1score_best_pre_l1reg = {}
true_positives_best_cond_l1reg = {}
true_positives_best_pre_l1reg = {}
precision_best_cond_l1reg = {}
precision_best_pre_l1reg = {}
recall_best_cond_l1reg = {}
recall_best_pre_l1reg = {}
cuts_best_cond_l1reg = {}
cuts_best_pre_l1reg = {}
cuts_l1reg_ALL = {}

ct_outer = 0

number_experiments = 0

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    eigv, lambda_val = fiedler_local(g, rr)
    lambda_val = np.real(lambda_val)
    
    step = (2*lambda_val - lambda_val/2)/4
    
    a_list = np.arange(lambda_val/2,2*lambda_val,step)
    
    ct = 0
    
    start = time.time()
    
    for node in nodes[ct_outer]:
        ref_node = [node]
        
        max_precision = -1
        min_conduct = 100
        
        ct_inner = 0
        for a in a_list:
            
            if ct_outer <= 1:
                rho = 0.15/np.sum(g.d[rr])
            else:
                rho = 0.2/np.sum(g.d[rr])
            
            output_pr_clustering = approximate_PageRank(g,ref_node,method = "l1reg-rand", epsilon=1.0e-2, rho=rho, alpha=a, cpp = True, normalize=True,normalized_objective=True)
            number_experiments += 1
            
            output_pr_sc = sweep_cut(g,output_pr_clustering,cpp=True)
            
            S = output_pr_sc[0]
            
#             cuts_l1reg_ALL[ct_outer,node,ct_inner] = S
            
            size_clust_l1reg_ = len(S)
            
            cond_val_l1pr = g.compute_conductance(S)
            
            vol_ = sum(g.d[S])
            true_positives_l1reg_ = set(rr).intersection(S)
            if len(true_positives_l1reg_) == 0:
                true_positives_l1reg_ = set(ref_node)
                vol_ = g.d[ref_node][0,0]
            precision = sum(g.d[np.array(list(true_positives_l1reg_))])/vol_
            recall = sum(g.d[np.array(list(true_positives_l1reg_))])/sum(g.d[rr])
            f1_score_ = 2*(precision*recall)/(precision + recall)
            
            if f1_score_ >= max_precision:
                
                max_precision = f1_score_
                
                external_best_pre_cond_l1reg[ct_outer,node] = cond_val_l1pr
                vol_best_pre_l1reg[ct_outer,node] = vol_
                
                size_clust_best_pre_l1reg[ct_outer,node] = size_clust_l1reg_
                true_positives_best_pre_l1reg[ct_outer,node] = true_positives_l1reg_
                precision_best_pre_l1reg[ct_outer,node] = precision
                recall_best_pre_l1reg[ct_outer,node] = recall
                f1score_best_pre_l1reg[ct_outer,node] = f1_score_
                
                cuts_best_pre_l1reg[ct_outer,node] = S
        
            if cond_val_l1pr <= min_conduct:
                
                min_conduct = cond_val_l1pr
                
                external_best_cond_l1reg[ct_outer,node] = cond_val_l1pr
                vol_best_cond_l1reg[ct_outer,node] = vol_
                
                size_clust_best_cond_l1reg[ct_outer,node] = size_clust_l1reg_
                true_positives_best_cond_l1reg[ct_outer,node] = true_positives_l1reg_
                precision_best_cond_l1reg[ct_outer,node] = precision
                recall_best_cond_l1reg[ct_outer,node] = recall
                f1score_best_cond_l1reg[ct_outer,node] = f1_score_
                
                cuts_best_cond_l1reg[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_l1reg[ct_outer,node], 'f1score: ', f1score_best_pre_l1reg[ct_outer,node], 'precision: ', precision_best_pre_l1reg[ct_outer,node], 'recall: ', recall_best_pre_l1reg[ct_outer,node])
        ct += 1

    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time l1-reg. with rounding: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1
    
## Performance of l1-reg. PR (with rounding).

all_data = []
xlabels_ = []

print('Results for l1-reg with rounding')
sum_precision = 0
sum_recall = 0
sum_f1 = 0
sum_conductance = 0

info_ref_nodes = all_clusters
l_info_ref_nodes = len(info_ref_nodes)

for i in range(l_info_ref_nodes):
    temp_pre = []
    temp_rec = []
    temp_f1 = []
    temp_conductance = []
    
    for j in all_clusters[i]:
        temp_pre.append(precision_best_cond_l1reg[i,j])
        temp_rec.append(recall_best_cond_l1reg[i,j])
        temp_f1.append(f1score_best_cond_l1reg[i,j])
        temp_conductance.append(external_best_cond_l1reg[i,j])
    
    print('Feature:', i,'Precision', stat_.mean(temp_pre), 'Recall', stat_.mean(temp_rec), 'F1', stat_.mean(temp_f1), 'Cond.', stat_.mean(temp_conductance))

np.save('f1score_best_pre_l1reg_ORKUT', f1score_best_pre_l1reg) 
np.save('precision_best_pre_l1reg_ORKUT', precision_best_pre_l1reg) 
np.save('recall_best_pre_l1reg_ORKUT', recall_best_pre_l1reg)

## Function for seed set expansion using BFS

import queue
def seed_grow_bfs_steps(g,seeds,steps,vol_target,target_cluster):
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
                    
                    vol_seeds = np.sum(g.d[seeds])
                    vol_target_intersection_input = np.sum(g.d[list(set(target_cluster).intersection(set(seeds)))])
                    sigma = vol_target_intersection_input/vol_target
                    
                    if sigma > 0.75 or vol_seeds > 0.25*g.vol_G:
                        break
                 
            vol_seeds = np.sum(g.d[seeds])
            vol_target_intersection_input = np.sum(g.d[list(set(target_cluster).intersection(set(seeds)))])
            sigma = vol_target_intersection_input/vol_target   
            
            if sigma > 0.75 or vol_seeds > 0.25*g.vol_G:
                break
               
        vol_seeds = np.sum(g.d[seeds])
        vol_target_intersection_input = np.sum(g.d[list(set(target_cluster).intersection(set(seeds)))])
        sigma = vol_target_intersection_input/vol_target
                
        if sigma > 0.75 or vol_seeds > 0.25*g.vol_G:
            break
    return seeds

## Collect data for seed set expansion + SL, try a lot of parameters

nodes = {}
external_best_cond_flBFS = {}
external_best_pre_cond_flBFS = {}
vol_best_cond_flBFS = {}
vol_best_pre_flBFS = {}
size_clust_best_cond_flBFS = {}
size_clust_best_pre_flBFS = {}
f1score_best_cond_flBFS = {}
f1score_best_pre_flBFS = {}
true_positives_best_cond_flBFS = {}
true_positives_best_pre_flBFS = {}
precision_best_cond_flBFS = {}
precision_best_pre_flBFS = {}
recall_best_cond_flBFS = {}
recall_best_pre_flBFS = {}
cuts_best_cond_flBFS = {}
cuts_best_pre_flBFS = {}
cuts_flBFS_ALL = {}

ct_outer = 0

number_experiments = 0

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    n_step = 24
    
    vol_target = np.sum(g.d[rr])
    
    ct = 0
    
    start = time.time()
    
    for node in nodes[ct_outer]:
        ref_node = [node]
        
        max_precision = -1
        min_conduct = 100
                
        seeds = seed_grow_bfs_steps(g,[node],g._num_vertices,vol_target,rr)

        vol_input = np.sum(g.d[seeds])

        vol_graph_minus_input = np.sum(g.d[list(set(range(g._num_vertices)) - set(seeds))])

        vol_target_intersection_input = np.sum(g.d[list(set(rr).intersection(set(seeds)))])

        gamma = vol_input/vol_graph_minus_input
                
        sigma = max(vol_target_intersection_input/vol_target,gamma)
        
        delta = min(max((1/3)*(1.0/(1.0/sigma - 1)) - gamma,0),1)
                        
        S = flow_clustering(g,seeds,method="sl",delta=delta)[0]
        number_experiments += 1

        cuts_flBFS_ALL[ct_outer,node] = S

        size_clust_flBFS_ = len(S)

        cond_val_l1pr = g.compute_conductance(S)

        vol_ = sum(g.d[S])
        true_positives_flBFS_ = set(rr).intersection(S)
        if len(true_positives_flBFS_) == 0:
            true_positives_flBFS_ = set(ref_node)
            vol_ = g.d[ref_node][0]
        precision = sum(g.d[np.array(list(true_positives_flBFS_))])/vol_
        recall = sum(g.d[np.array(list(true_positives_flBFS_))])/sum(g.d[rr])
        f1_score_ = 2*(precision*recall)/(precision + recall)

        if f1_score_ >= max_precision:

            max_precision = f1_score_

            external_best_pre_cond_flBFS[ct_outer,node] = cond_val_l1pr
            vol_best_pre_flBFS[ct_outer,node] = vol_

            size_clust_best_pre_flBFS[ct_outer,node] = size_clust_flBFS_
            true_positives_best_pre_flBFS[ct_outer,node] = true_positives_flBFS_
            precision_best_pre_flBFS[ct_outer,node] = precision
            recall_best_pre_flBFS[ct_outer,node] = recall
            f1score_best_pre_flBFS[ct_outer,node] = f1_score_

            cuts_best_pre_flBFS[ct_outer,node] = S

        if cond_val_l1pr <= min_conduct:

            min_conduct = cond_val_l1pr

            external_best_cond_flBFS[ct_outer,node] = cond_val_l1pr
            vol_best_cond_flBFS[ct_outer,node] = vol_

            size_clust_best_cond_flBFS[ct_outer,node] = size_clust_flBFS_
            true_positives_best_cond_flBFS[ct_outer,node] = true_positives_flBFS_
            precision_best_cond_flBFS[ct_outer,node] = precision
            recall_best_cond_flBFS[ct_outer,node] = recall
            f1score_best_cond_flBFS[ct_outer,node] = f1_score_

            cuts_best_cond_flBFS[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_flBFS[ct_outer,node], 'f1score: ', f1score_best_cond_flBFS[ct_outer,node], 'precision: ', precision_best_cond_flBFS[ct_outer,node], 'recall: ', recall_best_cond_flBFS[ct_outer,node])
        ct += 1
    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time BFS+SL: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1
    
## Performance of BFS+SL.

all_data = []
xlabels_ = []

print('Results for BFS+SL')
sum_precision = 0
sum_recall = 0
sum_f1 = 0
sum_conductance = 0

info_ref_nodes = all_clusters
l_info_ref_nodes = len(info_ref_nodes)

for i in range(l_info_ref_nodes):
    temp_pre = []
    temp_rec = []
    temp_f1 = []
    temp_conductance = []
    
    for j in all_clusters[i]:
        temp_pre.append(precision_best_cond_flBFS[i,j])
        temp_rec.append(recall_best_cond_flBFS[i,j])
        temp_f1.append(f1score_best_cond_flBFS[i,j])
        temp_conductance.append(external_best_cond_flBFS[i,j])

    print('Feature:', i,'Precision', stat_.mean(temp_pre), 'Recall', stat_.mean(temp_rec), 'F1', stat_.mean(temp_f1), 'Cond.', stat_.mean(temp_conductance))

np.save('f1score_best_pre_flBFS_ORKUT', f1score_best_pre_flBFS) 
np.save('precision_best_pre_flBFS_ORKUT', precision_best_pre_flBFS) 
np.save('recall_best_pre_flBFS_ORKUT', recall_best_pre_flBFS) 

## Collect data for APPR+SL
nodes = {}
external_best_cond_apprSL = {}
external_best_pre_cond_apprSL = {}
vol_best_cond_apprSL = {}
vol_best_pre_apprSL = {}
size_clust_best_cond_apprSL = {}
size_clust_best_pre_apprSL = {}
f1score_best_cond_apprSL = {}
f1score_best_pre_apprSL = {}
true_positives_best_cond_apprSL = {}
true_positives_best_pre_apprSL = {}
precision_best_cond_apprSL = {}
precision_best_pre_apprSL = {}
recall_best_cond_apprSL = {}
recall_best_pre_apprSL = {}
cuts_best_cond_apprSL = {}
cuts_best_pre_apprSL = {}
cuts_apprSL_ALL = {}

ct_outer = 0

number_experiments = 0

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    eigv, lambda_val = fiedler_local(g, rr)
    lambda_val = np.real(lambda_val)
    
    step = (2*lambda_val - lambda_val/2)/4
    
    a_list = np.arange(lambda_val/2,2*lambda_val,step)
    
    vol_target = np.sum(g.d[rr])
    
    ct = 0
    
    start = time.time()
    
    for node in nodes[ct_outer]:
        ref_node = [node]
        
        max_precision = -1
        min_conduct = 100
        
        ct_inner = 0
        for a in a_list:
            
            if ct_outer <= 1:
                rho = 0.15/np.sum(g.d[rr])
            else:
                rho = 0.2/np.sum(g.d[rr])
            
            output_pr_clustering = approximate_PageRank(g,ref_node,method = "acl", rho=rho, alpha=a, cpp = True, normalize=True,normalized_objective=True)
            number_experiments += 1
            
            output_pr_sc = sweep_cut(g,output_pr_clustering,cpp=True)
            
            S = output_pr_sc[0]
            
            vol_input = np.sum(g.d[S])

            vol_graph_minus_input = np.sum(g.d[list(set(range(g._num_vertices)) - set(S))])

            vol_target_intersection_input = np.sum(g.d[list(set(rr).intersection(set(S)))])

            gamma = vol_input/vol_graph_minus_input

            sigma = max(vol_target_intersection_input/vol_target,gamma)

            delta = min(max((1/3)*(1.0/(1.0/sigma - 1)) - gamma,0),1)

            S = flow_clustering(g,S,method="sl",delta=delta)[0]
            
            cuts_apprSL_ALL[ct_outer,node,ct_inner] = S
            
            size_clust_apprSL_ = len(S)
            
            cond_val_l1pr = g.compute_conductance(S)
            
            vol_ = sum(g.d[S])
            true_positives_apprSL_ = set(rr).intersection(S)
            if len(true_positives_apprSL_) == 0:
                true_positives_apprSL_ = set(ref_node)
                vol_ = g.d[ref_node][0]
            precision = sum(g.d[np.array(list(true_positives_apprSL_))])/vol_
            recall = sum(g.d[np.array(list(true_positives_apprSL_))])/sum(g.d[rr])
            f1_score_ = 2*(precision*recall)/(precision + recall)
            
            if f1_score_ >= max_precision:
                
                max_precision = f1_score_
                
                external_best_pre_cond_apprSL[ct_outer,node] = cond_val_l1pr
                vol_best_pre_apprSL[ct_outer,node] = vol_
                
                size_clust_best_pre_apprSL[ct_outer,node] = size_clust_apprSL_
                true_positives_best_pre_apprSL[ct_outer,node] = true_positives_apprSL_
                precision_best_pre_apprSL[ct_outer,node] = precision
                recall_best_pre_apprSL[ct_outer,node] = recall
                f1score_best_pre_apprSL[ct_outer,node] = f1_score_
                
                cuts_best_pre_apprSL[ct_outer,node] = S
        
            if cond_val_l1pr <= min_conduct:
                
                min_conduct = cond_val_l1pr
                
                external_best_cond_apprSL[ct_outer,node] = cond_val_l1pr
                vol_best_cond_apprSL[ct_outer,node] = vol_
                
                size_clust_best_cond_apprSL[ct_outer,node] = size_clust_apprSL_
                true_positives_best_cond_apprSL[ct_outer,node] = true_positives_apprSL_
                precision_best_cond_apprSL[ct_outer,node] = precision
                recall_best_cond_apprSL[ct_outer,node] = recall
                f1score_best_cond_apprSL[ct_outer,node] = f1_score_
                
                cuts_best_cond_apprSL[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_apprSL[ct_outer,node], 'f1score: ', f1score_best_cond_apprSL[ct_outer,node], 'precision: ', precision_best_cond_apprSL[ct_outer,node], 'recall: ', recall_best_cond_apprSL[ct_outer,node])
        ct += 1
    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time APPR+SL with rounding: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1

## Performance of APPR+SL

all_data = []
xlabels_ = []

print('Results for APPR+SL')
sum_precision = 0
sum_recall = 0
sum_f1 = 0
sum_conductance = 0

info_ref_nodes = all_clusters
l_info_ref_nodes = len(info_ref_nodes)

for i in range(l_info_ref_nodes):
    temp_pre = []
    temp_rec = []
    temp_f1 = []
    temp_conductance = []
    
    for j in all_clusters[i]:
        temp_pre.append(precision_best_cond_apprSL[i,j])
        temp_rec.append(recall_best_cond_apprSL[i,j])
        temp_f1.append(f1score_best_cond_apprSL[i,j])
        temp_conductance.append(external_best_cond_apprSL[i,j])

    print('Feature:', i,'Precision', stat_.mean(temp_pre), 'Recall', stat_.mean(temp_rec), 'F1', stat_.mean(temp_f1), 'Cond.', stat_.mean(temp_conductance))


np.save('f1score_best_pre_apprSL_ORKUT', f1score_best_pre_apprSL) 
np.save('precision_best_pre_apprSL_ORKUT', precision_best_pre_apprSL) 
np.save('recall_best_pre_apprSL_ORKUT', recall_best_pre_apprSL) 

## Collect data for L1+SL

nodes = {}
external_best_cond_l1SL = {}
external_best_pre_cond_l1SL = {}
vol_best_cond_l1SL = {}
vol_best_pre_l1SL = {}
size_clust_best_cond_l1SL = {}
size_clust_best_pre_l1SL = {}
f1score_best_cond_l1SL = {}
f1score_best_pre_l1SL = {}
true_positives_best_cond_l1SL = {}
true_positives_best_pre_l1SL = {}
precision_best_cond_l1SL = {}
precision_best_pre_l1SL = {}
recall_best_cond_l1SL = {}
recall_best_pre_l1SL = {}
cuts_best_cond_l1SL = {}
cuts_best_pre_l1SL = {}
cuts_l1SL_ALL = {}

ct_outer = 0

number_experiments = 0

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    eigv, lambda_val = fiedler_local(g, rr)
    lambda_val = np.real(lambda_val)
    
    step = (2*lambda_val - lambda_val/2)/4
    
    a_list = np.arange(lambda_val/2,2*lambda_val,step)
    
    vol_target = np.sum(g.d[rr])
    
    ct = 0
    
    start = time.time()
    
    for node in nodes[ct_outer]:
        ref_node = [node]
        
        max_precision = -1
        min_conduct = 100
        
        ct_inner = 0
        for a in a_list:
            
            if ct_outer <= 1:
                rho = 0.15/np.sum(g.d[rr])
            else:
                rho = 0.2/np.sum(g.d[rr])
            
            output_pr_clustering = approximate_PageRank(g,ref_node,method = "l1reg-rand", epsilon=1.0e-2, rho=rho, alpha=a, cpp = True, normalize=True,normalized_objective=True)
            number_experiments += 1
            
            output_pr_sc = sweep_cut(g,output_pr_clustering,cpp=True)
            
            S = output_pr_sc[0]
            
            vol_input = np.sum(g.d[S])

            vol_graph_minus_input = np.sum(g.d[list(set(range(g._num_vertices)) - set(S))])

            vol_target_intersection_input = np.sum(g.d[list(set(rr).intersection(set(S)))])

            gamma = vol_input/vol_graph_minus_input

            sigma = max(vol_target_intersection_input/vol_target,gamma)

            delta = min(max((1/3)*(1.0/(1.0/sigma - 1)) - gamma,0),1)

            S = flow_clustering(g,S,method="sl",delta=delta)[0]
            
            cuts_l1SL_ALL[ct_outer,node,ct_inner] = S
            
            size_clust_l1SL_ = len(S)
            
            cond_val_l1pr = g.compute_conductance(S)
            
            vol_ = sum(g.d[S])
            true_positives_l1SL_ = set(rr).intersection(S)
            if len(true_positives_l1SL_) == 0:
                true_positives_l1SL_ = set(ref_node)
                vol_ = g.d[ref_node][0]
            precision = sum(g.d[np.array(list(true_positives_l1SL_))])/vol_
            recall = sum(g.d[np.array(list(true_positives_l1SL_))])/sum(g.d[rr])
            f1_score_ = 2*(precision*recall)/(precision + recall)
            
            if f1_score_ >= max_precision:
                
                max_precision = f1_score_
                
                external_best_pre_cond_l1SL[ct_outer,node] = cond_val_l1pr
                vol_best_pre_l1SL[ct_outer,node] = vol_
                
                size_clust_best_pre_l1SL[ct_outer,node] = size_clust_l1SL_
                true_positives_best_pre_l1SL[ct_outer,node] = true_positives_l1SL_
                precision_best_pre_l1SL[ct_outer,node] = precision
                recall_best_pre_l1SL[ct_outer,node] = recall
                f1score_best_pre_l1SL[ct_outer,node] = f1_score_
                
                cuts_best_pre_l1SL[ct_outer,node] = S
        
            if cond_val_l1pr <= min_conduct:
                
                min_conduct = cond_val_l1pr
                
                external_best_cond_l1SL[ct_outer,node] = cond_val_l1pr
                vol_best_cond_l1SL[ct_outer,node] = vol_
                
                size_clust_best_cond_l1SL[ct_outer,node] = size_clust_l1SL_
                true_positives_best_cond_l1SL[ct_outer,node] = true_positives_l1SL_
                precision_best_cond_l1SL[ct_outer,node] = precision
                recall_best_cond_l1SL[ct_outer,node] = recall
                f1score_best_cond_l1SL[ct_outer,node] = f1_score_
                
                cuts_best_cond_l1SL[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_l1SL[ct_outer,node], 'f1score: ', f1score_best_cond_l1SL[ct_outer,node], 'precision: ', precision_best_cond_l1SL[ct_outer,node], 'recall: ', recall_best_cond_l1SL[ct_outer,node])
        ct += 1
    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time L1+SL with rounding: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1
    
## Performance of l1+SL

all_data = []
xlabels_ = []

print('Results for L1+SL')
sum_precision = 0
sum_recall = 0
sum_f1 = 0
sum_conductance = 0

info_ref_nodes = all_clusters
l_info_ref_nodes = len(info_ref_nodes)

for i in range(l_info_ref_nodes):
    temp_pre = []
    temp_rec = []
    temp_f1 = []
    temp_conductance = []
    
    for j in all_clusters[i]:
        temp_pre.append(precision_best_cond_l1SL[i,j])
        temp_rec.append(recall_best_cond_l1SL[i,j])
        temp_f1.append(f1score_best_cond_l1SL[i,j])
        temp_conductance.append(external_best_cond_l1SL[i,j])

    print('Feature:', i,'Precision', stat_.mean(temp_pre), 'Recall', stat_.mean(temp_rec), 'F1', stat_.mean(temp_f1), 'Cond.', stat_.mean(temp_conductance))
    
np.save('f1score_best_pre_l1SL_ORKUT', f1score_best_pre_l1SL) 
np.save('precision_best_pre_l1SL_ORKUT', precision_best_pre_l1SL) 
np.save('recall_best_pre_l1SL_ORKUT', recall_best_pre_l1SL)
    