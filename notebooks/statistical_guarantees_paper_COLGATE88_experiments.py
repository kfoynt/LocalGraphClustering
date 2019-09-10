import scipy as sp
import numpy as np
import time

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

g = GraphLocal('datasets/Colgate88_reduced.graphml','graphml')

G = nx.read_graphml('datasets/Colgate88_reduced.graphml')

ground_truth_clusters_by_number = dict()

cluster_names = ['secondMajor','highSchool','gender','dorm','majorIndex','year']

for node in G.nodes(data=True):
    
    for cluster_name in cluster_names:
        
        ground_truth_clusters_by_number[cluster_name+str(node[1][cluster_name])] =  []

for node in G.nodes(data=True):
    
    for cluster_name in cluster_names:
        
        ground_truth_clusters_by_number[cluster_name+str(node[1][cluster_name])].append(int(node[0]))
        
all_clusters = []
counter = 0
for cluster_id in ground_truth_clusters_by_number:
    
    cluster = ground_truth_clusters_by_number[cluster_id]
    
    if len(cluster) == 1 or len(cluster) == 0:
        counter += 1
        continue
    
#     eig, lambda_ = fiedler_local(g, cluster)
#     lambda_ = np.real(lambda_)
#     gap = lambda_/g.compute_conductance(cluster)
    cond = g.compute_conductance(cluster)
    counter += 1
    
    if cond <= 0.57 and len(cluster) >= 10:
        print("Id: ", cluster_id)
        print("Cluster: ", counter, " conductance: ", cond, "Size: ", len(cluster), " Volume: ", np.sum(g.d[cluster]))
        all_clusters.append(cluster)

## Collect data of ACL (without rounding)

nodes = {}
external_best_cond_acl = {}
external_best_pre_cond_acl = {}
gap_best_cond_acl = {}
gap_best_pre_acl = {}
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

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    eigv, lambda_val = fiedler_local(g, rr)
    lambda_val = np.real(lambda_val)
    
    step = (2*lambda_val - lambda_val/2)/4
    
    a_list = np.arange(lambda_val/2,2*lambda_val,step)
    
    start = time.time()
    number_experiments = 0
    
    ct = 0
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
            
            S = output_pr_clustering[0]
            
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
                
                S_smqi, S_smqi_val = fiedler_local(g, S)
                S_smqi_val = np.real(S_smqi_val)
                
                external_best_pre_cond_acl[ct_outer,node] = cond_val_l1pr
                gap_best_pre_acl[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                vol_best_pre_acl[ct_outer,node] = vol_
                
                size_clust_best_pre_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_pre_acl[ct_outer,node] = true_positives_acl_
                precision_best_pre_acl[ct_outer,node] = precision
                recall_best_pre_acl[ct_outer,node] = recall
                f1score_best_pre_acl[ct_outer,node] = f1_score_
                
                cuts_best_pre_acl[ct_outer,node] = S
        
            if cond_val_l1pr <= min_conduct:
                
                min_conduct = cond_val_l1pr
                
                S_smqi, S_smqi_val = fiedler_local(g, S)
                S_smqi_val = np.real(S_smqi_val)
                
                external_best_cond_acl[ct_outer,node] = cond_val_l1pr
                gap_best_cond_acl[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                vol_best_cond_acl[ct_outer,node] = vol_
                
                size_clust_best_cond_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_cond_acl[ct_outer,node] = true_positives_acl_
                precision_best_cond_acl[ct_outer,node] = precision
                recall_best_cond_acl[ct_outer,node] = recall
                f1score_best_cond_acl[ct_outer,node] = f1_score_
                
                cuts_best_cond_acl[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_acl[ct_outer,node], 'f1score: ', f1score_best_pre_acl[ct_outer,node], 'precision: ', precision_best_pre_acl[ct_outer,node], 'recall: ', recall_best_pre_acl[ct_outer,node])
        ct += 1

    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time ACL without rounding: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1

## Performance of ACL (without rounding).

all_data = []
xlabels_ = []

print('Results for ACL without rounding')
sum_precision = 0
sum_recall = 0
sum_f1 = 0
sum_conductance = 0

info_ref_nodes = nodes
l_info_ref_nodes = len(info_ref_nodes)

for i in range(l_info_ref_nodes):
    temp_pre = []
    temp_rec = []
    temp_f1 = []
    temp_conductance = []
    
    for j in nodes[i]:
        temp_pre.append(precision_best_cond_acl[i,j])
        temp_rec.append(recall_best_cond_acl[i,j])
        temp_f1.append(f1score_best_cond_acl[i,j])
        temp_conductance.append(external_best_cond_acl[i,j])
    
    print('Feature:', i,'Precision', stat_.median(temp_pre), 'Recall', stat_.median(temp_rec), 'F1', stat_.median(temp_f1), 'Cond.', stat_.median(temp_conductance))
    sum_precision += stat_.median(temp_pre)
    sum_recall += stat_.median(temp_rec)
    sum_f1 += stat_.median(temp_f1)
    sum_conductance += stat_.median(temp_conductance)

avg_precision = sum_precision/l_info_ref_nodes
avg_recall = sum_recall/l_info_ref_nodes
avg_f1 = sum_f1/l_info_ref_nodes
avg_conductance = sum_conductance/l_info_ref_nodes

## Collect data for ACL (with rounding)

nodes = {}
external_best_cond_acl = {}
external_best_pre_cond_acl = {}
gap_best_cond_acl = {}
gap_best_pre_acl = {}
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

start = time.time()

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
                
                S_smqi, S_smqi_val = fiedler_local(g, S)
                S_smqi_val = np.real(S_smqi_val)
                
                external_best_pre_cond_acl[ct_outer,node] = cond_val_l1pr
                gap_best_pre_acl[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                vol_best_pre_acl[ct_outer,node] = vol_
                
                size_clust_best_pre_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_pre_acl[ct_outer,node] = true_positives_acl_
                precision_best_pre_acl[ct_outer,node] = precision
                recall_best_pre_acl[ct_outer,node] = recall
                f1score_best_pre_acl[ct_outer,node] = f1_score_
                
                cuts_best_pre_acl[ct_outer,node] = S
        
            if cond_val_l1pr <= min_conduct:
                
                min_conduct = cond_val_l1pr
                
                S_smqi, S_smqi_val = fiedler_local(g, S)
                S_smqi_val = np.real(S_smqi_val)
                
                external_best_cond_acl[ct_outer,node] = cond_val_l1pr
                gap_best_cond_acl[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                vol_best_cond_acl[ct_outer,node] = vol_
                
                size_clust_best_cond_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_cond_acl[ct_outer,node] = true_positives_acl_
                precision_best_cond_acl[ct_outer,node] = precision
                recall_best_cond_acl[ct_outer,node] = recall
                f1score_best_cond_acl[ct_outer,node] = f1_score_
                
                cuts_best_cond_acl[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_acl[ct_outer,node], 'f1score: ', f1score_best_pre_acl[ct_outer,node], 'precision: ', precision_best_pre_acl[ct_outer,node], 'recall: ', recall_best_pre_acl[ct_outer,node])
        ct += 1
    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time ACL without rounding: ", end - start)
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
    
    print('Feature:', i,'Precision', stat_.median(temp_pre), 'Recall', stat_.median(temp_rec), 'F1', stat_.median(temp_f1), 'Cond.', stat_.median(temp_conductance))
    sum_precision += stat_.median(temp_pre)
    sum_recall += stat_.median(temp_rec)
    sum_f1 += stat_.median(temp_f1)
    sum_conductance += stat_.median(temp_conductance)

avg_precision = sum_precision/l_info_ref_nodes
avg_recall = sum_recall/l_info_ref_nodes
avg_f1 = sum_f1/l_info_ref_nodes
avg_conductance = sum_conductance/l_info_ref_nodes

## Collect data for l1-reg. PR (without rounding)

nodes = {}
external_best_cond_acl = {}
external_best_pre_cond_acl = {}
gap_best_cond_acl = {}
gap_best_pre_acl = {}
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

start = time.time()

number_experiments = 0

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    #     nodes[ct_outer] = rr
    
    eigv, lambda_val = fiedler_local(g, rr)
    lambda_val = np.real(lambda_val)
    
    step = (2*lambda_val - lambda_val/2)/4
    
    a_list = np.arange(lambda_val/2,2*lambda_val,step)
    
    ct = 0
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
            
            S = output_pr_clustering[0]
            
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
                
                S_smqi, S_smqi_val = fiedler_local(g, S)
                S_smqi_val = np.real(S_smqi_val)
                
                external_best_pre_cond_acl[ct_outer,node] = cond_val_l1pr
                gap_best_pre_acl[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                vol_best_pre_acl[ct_outer,node] = vol_
                
                size_clust_best_pre_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_pre_acl[ct_outer,node] = true_positives_acl_
                precision_best_pre_acl[ct_outer,node] = precision
                recall_best_pre_acl[ct_outer,node] = recall
                f1score_best_pre_acl[ct_outer,node] = f1_score_
                
                cuts_best_pre_acl[ct_outer,node] = S
        
            if cond_val_l1pr <= min_conduct:
                
                min_conduct = cond_val_l1pr
                
                S_smqi, S_smqi_val = fiedler_local(g, S)
                S_smqi_val = np.real(S_smqi_val)
                
                external_best_cond_acl[ct_outer,node] = cond_val_l1pr
                gap_best_cond_acl[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                vol_best_cond_acl[ct_outer,node] = vol_
                
                size_clust_best_cond_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_cond_acl[ct_outer,node] = true_positives_acl_
                precision_best_cond_acl[ct_outer,node] = precision
                recall_best_cond_acl[ct_outer,node] = recall
                f1score_best_cond_acl[ct_outer,node] = f1_score_
                
                cuts_best_cond_acl[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_acl[ct_outer,node], 'f1score: ', f1score_best_pre_acl[ct_outer,node], 'precision: ', precision_best_pre_acl[ct_outer,node], 'recall: ', recall_best_pre_acl[ct_outer,node])
        ct += 1

    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time ACL without rounding: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1

## Performance of l1-reg. PR (without rounding).

all_data = []
xlabels_ = []

print('Results for l1-reg without rounding')
sum_precision = 0
sum_recall = 0
sum_f1 = 0
sum_conductance = 0

info_ref_nodes = nodes
l_info_ref_nodes = len(info_ref_nodes)

for i in range(l_info_ref_nodes):
    temp_pre = []
    temp_rec = []
    temp_f1 = []
    temp_conductance = []
    
    for j in nodes[i]:
        temp_pre.append(precision_best_cond_acl[i,j])
        temp_rec.append(recall_best_cond_acl[i,j])
        temp_f1.append(f1score_best_cond_acl[i,j])
        temp_conductance.append(external_best_cond_acl[i,j])
    
    print('Feature:', i,'Precision', stat_.median(temp_pre), 'Recall', stat_.median(temp_rec), 'F1', stat_.median(temp_f1), 'Cond.', stat_.median(temp_conductance))
    sum_precision += stat_.median(temp_pre)
    sum_recall += stat_.median(temp_rec)
    sum_f1 += stat_.median(temp_f1)
    sum_conductance += stat_.median(temp_conductance)

avg_precision = sum_precision/l_info_ref_nodes
avg_recall = sum_recall/l_info_ref_nodes
avg_f1 = sum_f1/l_info_ref_nodes
avg_conductance = sum_conductance/l_info_ref_nodes

## Collect data for l1-reg. PR (with rounding)

nodes = {}
external_best_cond_acl = {}
external_best_pre_cond_acl = {}
gap_best_cond_acl = {}
gap_best_pre_acl = {}
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

start = time.time()

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
                
                S_smqi, S_smqi_val = fiedler_local(g, S)
                S_smqi_val = np.real(S_smqi_val)
                
                external_best_pre_cond_acl[ct_outer,node] = cond_val_l1pr
                gap_best_pre_acl[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                vol_best_pre_acl[ct_outer,node] = vol_
                
                size_clust_best_pre_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_pre_acl[ct_outer,node] = true_positives_acl_
                precision_best_pre_acl[ct_outer,node] = precision
                recall_best_pre_acl[ct_outer,node] = recall
                f1score_best_pre_acl[ct_outer,node] = f1_score_
                
                cuts_best_pre_acl[ct_outer,node] = S
        
            if cond_val_l1pr <= min_conduct:
                
                min_conduct = cond_val_l1pr
                
                S_smqi, S_smqi_val = fiedler_local(g, S)
                S_smqi_val = np.real(S_smqi_val)
                
                external_best_cond_acl[ct_outer,node] = cond_val_l1pr
                gap_best_cond_acl[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                vol_best_cond_acl[ct_outer,node] = vol_
                
                size_clust_best_cond_acl[ct_outer,node] = size_clust_acl_
                true_positives_best_cond_acl[ct_outer,node] = true_positives_acl_
                precision_best_cond_acl[ct_outer,node] = precision
                recall_best_cond_acl[ct_outer,node] = recall
                f1score_best_cond_acl[ct_outer,node] = f1_score_
                
                cuts_best_cond_acl[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_acl[ct_outer,node], 'f1score: ', f1score_best_pre_acl[ct_outer,node], 'precision: ', precision_best_pre_acl[ct_outer,node], 'recall: ', recall_best_pre_acl[ct_outer,node])
        ct += 1

    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time ACL without rounding: ", end - start)
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
        temp_pre.append(precision_best_cond_acl[i,j])
        temp_rec.append(recall_best_cond_acl[i,j])
        temp_f1.append(f1score_best_cond_acl[i,j])
        temp_conductance.append(external_best_cond_acl[i,j])
    
    print('Feature:', i,'Precision', stat_.median(temp_pre), 'Recall', stat_.median(temp_rec), 'F1', stat_.median(temp_f1), 'Cond.', stat_.median(temp_conductance))
    sum_precision += stat_.median(temp_pre)
    sum_recall += stat_.median(temp_rec)
    sum_f1 += stat_.median(temp_f1)
    sum_conductance += stat_.median(temp_conductance)

avg_precision = sum_precision/l_info_ref_nodes
avg_recall = sum_recall/l_info_ref_nodes
avg_f1 = sum_f1/l_info_ref_nodes
avg_conductance = sum_conductance/l_info_ref_nodes

## Function for seed set expansion using BFS

import queue
def seed_grow_bfs_steps(g,seeds,steps,vol_target):
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
                    if np.sum(g.d[seeds]) > 0.75*vol_target:
                        break
            if np.sum(g.d[seeds]) > 0.75*vol_target:
                break
        if np.sum(g.d[seeds]) > 0.75*vol_target:
            break
    return seeds

## Collect data for seed set expansion + FlowImprove, try a lot of parameters

nodes = {}
external_best_cond_flBFS = {}
external_best_pre_cond_flBFS = {}
gap_best_cond_flBFS = {}
gap_best_pre_flBFS = {}
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

start = time.time()

number_experiments = 0

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    eigv, lambda_val = fiedler_local(g, rr)
    lambda_val = np.real(lambda_val)
    
    n_step = 24
    
    step_list = range(1,n_step,5)
    
    delta_list = [1.0e-8,0.3,0.6,1]
    
    ct = 0
    for node in nodes[ct_outer]:
        ref_node = [node]
        
        max_precision = -1
        min_conduct = 100
        
        ct_inner = 0
        for steps in step_list:
            
            ct_double_inner = 0
            
            for delta in delta_list:
                
                seeds = seed_grow_bfs_steps(g,[node],steps,np.sum(g.d[rr]))
                
                S = flow_clustering(g,seeds,method="sl",delta=delta)[0]
                number_experiments += 1
                
                cuts_flBFS_ALL[ct_outer,node,ct_inner,ct_double_inner] = S
                
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
                    
                    S_smqi, S_smqi_val = fiedler_local(g, S)
                    S_smqi_val = np.real(S_smqi_val)
                    
                    external_best_pre_cond_flBFS[ct_outer,node] = cond_val_l1pr
                    gap_best_pre_flBFS[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                    vol_best_pre_flBFS[ct_outer,node] = vol_
                    
                    size_clust_best_pre_flBFS[ct_outer,node] = size_clust_flBFS_
                    true_positives_best_pre_flBFS[ct_outer,node] = true_positives_flBFS_
                    precision_best_pre_flBFS[ct_outer,node] = precision
                    recall_best_pre_flBFS[ct_outer,node] = recall
                    f1score_best_pre_flBFS[ct_outer,node] = f1_score_
                    
                    cuts_best_pre_flBFS[ct_outer,node] = S
            
                if cond_val_l1pr <= min_conduct:
                    
                    min_conduct = cond_val_l1pr
                    
                    S_smqi, S_smqi_val = fiedler_local(g, S)
                    S_smqi_val = np.real(S_smqi_val)
                    
                    external_best_cond_flBFS[ct_outer,node] = cond_val_l1pr
                    gap_best_cond_flBFS[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                    vol_best_cond_flBFS[ct_outer,node] = vol_
                    
                    size_clust_best_cond_flBFS[ct_outer,node] = size_clust_flBFS_
                    true_positives_best_cond_flBFS[ct_outer,node] = true_positives_flBFS_
                    precision_best_cond_flBFS[ct_outer,node] = precision
                    recall_best_cond_flBFS[ct_outer,node] = recall
                    f1score_best_cond_flBFS[ct_outer,node] = f1_score_
                    
                    cuts_best_cond_flBFS[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_flBFS[ct_outer,node], 'f1score: ', f1score_best_pre_flBFS[ct_outer,node], 'precision: ', precision_best_pre_flBFS[ct_outer,node], 'recall: ', recall_best_pre_flBFS[ct_outer,node])
        ct += 1
    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time ACL without rounding: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1

## Performance of BFS+FlowImp.

all_data = []
xlabels_ = []

print('Results for BFS+FlowImp')
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
    
    print('Feature:', i,'Precision', stat_.median(temp_pre), 'Recall', stat_.median(temp_rec), 'F1', stat_.median(temp_f1), 'Cond.', stat_.median(temp_conductance))
    sum_precision += stat_.median(temp_pre)
    sum_recall += stat_.median(temp_rec)
    sum_f1 += stat_.median(temp_f1)
    sum_conductance += stat_.median(temp_conductance)

avg_precision = sum_precision/l_info_ref_nodes
avg_recall = sum_recall/l_info_ref_nodes
avg_f1 = sum_f1/l_info_ref_nodes
avg_conductance = sum_conductance/l_info_ref_nodes

## Collect data for seed set expansion + FlowImprove, as many parameters as l1-reg

nodes = {}
external_best_cond_flBFS = {}
external_best_pre_cond_flBFS = {}
gap_best_cond_flBFS = {}
gap_best_pre_flBFS = {}
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

start = time.time()

number_experiments = 0

for rr in all_clusters:
    
    how_many = int(len(rr))
    print(how_many)
    
    random.seed(4)
    
    nodes[ct_outer] = np.random.choice(rr, how_many, replace=False)
    
    eigv, lambda_val = fiedler_local(g, rr)
    lambda_val = np.real(lambda_val)
    
    step_list = range(1,12,5)
    
    delta_list = [1.0e-8,0.3,0.6,1]
    
    ct = 0
    for node in nodes[ct_outer]:
        ref_node = [node]
        
        max_precision = -1
        min_conduct = 100
        
        ct_inner = 0
        for steps in step_list:
            
            ct_double_inner = 0
            
            for delta in delta_list:
                
                seeds = seed_grow_bfs_steps(g,[node],steps,np.sum(g.d[rr]))
                
                S = flow_clustering(g,seeds,method="sl",delta=delta)[0]
                number_experiments += 1
                
                cuts_flBFS_ALL[ct_outer,node,ct_inner,ct_double_inner] = S
                
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
                    
                    S_smqi, S_smqi_val = fiedler_local(g, S)
                    S_smqi_val = np.real(S_smqi_val)
                    
                    external_best_pre_cond_flBFS[ct_outer,node] = cond_val_l1pr
                    gap_best_pre_flBFS[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                    vol_best_pre_flBFS[ct_outer,node] = vol_
                    
                    size_clust_best_pre_flBFS[ct_outer,node] = size_clust_flBFS_
                    true_positives_best_pre_flBFS[ct_outer,node] = true_positives_flBFS_
                    precision_best_pre_flBFS[ct_outer,node] = precision
                    recall_best_pre_flBFS[ct_outer,node] = recall
                    f1score_best_pre_flBFS[ct_outer,node] = f1_score_
                    
                    cuts_best_pre_flBFS[ct_outer,node] = S
            
                if cond_val_l1pr <= min_conduct:
                    
                    min_conduct = cond_val_l1pr
                    
                    S_smqi, S_smqi_val = fiedler_local(g, S)
                    S_smqi_val = np.real(S_smqi_val)
                    
                    external_best_cond_flBFS[ct_outer,node] = cond_val_l1pr
                    gap_best_cond_flBFS[ct_outer,node] = (S_smqi_val/np.log(sum(g.d[S])))/cond_val_l1pr
                    vol_best_cond_flBFS[ct_outer,node] = vol_
                    
                    size_clust_best_cond_flBFS[ct_outer,node] = size_clust_flBFS_
                    true_positives_best_cond_flBFS[ct_outer,node] = true_positives_flBFS_
                    precision_best_cond_flBFS[ct_outer,node] = precision
                    recall_best_cond_flBFS[ct_outer,node] = recall
                    f1score_best_cond_flBFS[ct_outer,node] = f1_score_
                    
                    cuts_best_cond_flBFS[ct_outer,node] = S

        print('outer:', ct_outer, 'number of node: ',node, ' completed: ', ct/how_many, ' degree: ', g.d[node])
        print('conductance: ', external_best_cond_flBFS[ct_outer,node], 'f1score: ', f1score_best_pre_flBFS[ct_outer,node], 'precision: ', precision_best_pre_flBFS[ct_outer,node], 'recall: ', recall_best_pre_flBFS[ct_outer,node])
        ct += 1
    end = time.time()
    print(" ")
    print("Outer: ", ct_outer," Elapsed time ACL without rounding: ", end - start)
    print("Outer: ", ct_outer," Number of experiments: ", number_experiments)
    print(" ")
    ct_outer += 1

## Performance of BFS+FlowImp.

all_data = []
xlabels_ = []

print('Results for BFS+FlowImp')
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
    
    print('Feature:', i,'Precision', stat_.median(temp_pre), 'Recall', stat_.median(temp_rec), 'F1', stat_.median(temp_f1), 'Cond.', stat_.median(temp_conductance))
    sum_precision += stat_.median(temp_pre)
    sum_recall += stat_.median(temp_rec)
    sum_f1 += stat_.median(temp_f1)
    sum_conductance += stat_.median(temp_conductance)

avg_precision = sum_precision/l_info_ref_nodes
avg_recall = sum_recall/l_info_ref_nodes
avg_f1 = sum_f1/l_info_ref_nodes
avg_conductance = sum_conductance/l_info_ref_nodes
