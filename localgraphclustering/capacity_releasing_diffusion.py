import numpy as np
from unit_flow import unit_flow
from scipy import sparse as sp

def capacity_releasing_diffusion(ref_node,A,U=3,h=10,w=2,iterations=20, vol_G = []):
    """Description
       -----------
       
       Algorithm Capacity Releasing Diffusion for local graph clustering. This algorithm uses 
       a flow based method to push excess flow out of nodes. The algorithm is in worst-case 
       faster and stays more local than classical spectral diffusion processes.
       For more details please refere to: D. Wang, K. Fountoulakis, M. Henzinger, M. Mahoney 
       and S. Rao. Capacity Releasing Diffusion for Speed and Locality. ICML 2017.
       arXiv link: https://arxiv.org/abs/1706.05826
       
       Standard call
       -------------
       
       cut = excess_unit_flow(ref_node,A,U=3,h=10,w=2,iterations=20)
       
       Data input (mandatory)
       -----------------
       
       ref_node:  integer
                  The reference node, i.e., node of interest around which
                  we are looking for a target cluster.
                  
       A:         float, double
                  Compressed Sparse Row (CSR) symmetric matrix
                  The adjacency matrix that stores the connectivity of the graph.
                  For this algorithm the graph must be undirected and unweighted,
                  which means that matrix A must be symmetric and its elements 
                  are equal to either zero or one.
       
       Algorithm parameters (optional)
       -----------------
      
       U: integer
          default == 3
          The net mass any edge can be at most.
          
       h: integer
          defaul == 10
          The label of any node can have at most.
          
       w: integer
          default == 2
          Multiplicative factor for increasing the capacity of the nodes at each iteration.
          
       iterations: integer
                   default = 20
                   Maximum number of iterations of Capacity Releasing Diffusion Algorithm.
          
       For details of these parameters please refer to: D. Wang, K. Fountoulakis, M. Henzinger, 
       M. Mahoney and S. Rao. Capacity Releasing Diffusion for Speed and Locality. ICML 2017.
       arXiv link: https://arxiv.org/abs/1706.05826
       
       Output
       ------
       
       cut:  list
             A list of nodes that correspond to the cluster with the best 
             conductance that was found by the Capacity Releasing Diffusion Algorithm.
       
       Printing statements (warnings)
       ------------------------------
       
       Too much excess: Meanss that push/relabel cannot push the excess flow out of the nodes.
                        This might indicate that a cluster has been found. In this case the best
                        cluster in terms of conductance is returned.
                       
       Too much flow:   Means that the algorithm has touched about a third of the whole given graph.
                        The algorithm is terminated in this case and the best cluster in terms of 
                        conductance is returned. 
    """
    [n,n] = A.shape
    if vol_G == []:
        vol_G = sum(A.sum(axis=1))
    degree = {}

    Delta = {}
    Delta0 = {}
    for i in ref_node:
        if not(degree.has_key(i)):
            degree.update({i:A[i,:].sum(axis=1)[0,0]})
        degree_val = degree[i]
        Delta.update({i:2*degree_val})
        Delta0.update({i:2*degree_val})
        
    cond_best = 100
    
    for i in range(iterations):
        l,f_v,ex = unit_flow(A, Delta, U, h, w, degree)
        
        cond_temp, labels_temp = round_unit_flow(A,n,l,vol_G,degree)
        
        idx = min(cond_temp, key=cond_temp.get)
        
        if cond_temp[idx] < cond_best:
            cond_best = cond_temp[idx]
            cond_best_array = cond_temp
            labels = labels_temp
            
        total_excess = 0
        for j in f_v:
            if not(ex.has_key(j)):
                ex[j]=0
            total_excess += ex[j]
            Delta.update({j:w*(f_v[j] - ex[j])})
            
            
        if not(degree.has_key(ref_node[0])):
            degree.update({ref_node[0]:A[ref_node[0],:].sum(axis=1)[0,0]})
        degree_val = degree[ref_node[0]]         
        if (total_excess > (degree_val*np.exp2(i)/10)):
            print('Too much excess.', 'iteration:', i)
            break
        
        sum_ = 0
        for ttt in f_v.keys():
            if not(degree.has_key(ttt)):
                degree.update({ttt:A[ttt,:].sum(axis=1)[0,0]})
            degree_val = degree[ttt]  
            if f_v[ttt] >= degree_val:
                sum_ += degree_val
        
        if sum_ > vol_G/3:
            print('Too much flow.', 'iteration:', i)
            break
    
    cut = []
    idx = min(cond_best_array, key=cond_best_array.get)
    for e in reversed(sorted(cond_best_array.keys())):
        if e >= idx:
            for ee in labels[e]:
                cut.append(ee)
    
    return cut

def round_unit_flow(A,n,l,vol_G,degree):
    
    labels = {}
    for i in l:
        if not(labels.has_key(l[i])):
            labels.update({l[i]:list([i])})
        else:
            temp = labels[l[i]]
            temp.append(i)
            labels.update({l[i]:temp}) 
    
    cut = {}
    temp_prev = sp.csr_matrix((n,1),dtype=int)
    A_temp_prev = sp.csr_matrix((n,1),dtype=int)
    vol = {}
    vol_sum = 0
    quad_prev = 0
    
    for i in reversed(sorted(labels.keys())):
        temp_new = sp.csr_matrix((n,1),dtype=int)
        if i == 0:
            continue
        else:
            temp_new[labels[i]] = 1
            
            for j in labels[i]:
                if not(degree.has_key(j)):
                    degree.update({j:A[j,:].sum(axis=1)[0,0]})
                vol_sum += degree[j]
            vol.update({i:vol_sum}) 
            
            quad_new = temp_new.T.dot(A.dot(temp_new))[0,0]
            quad_prev_new = temp_new.T.dot(A_temp_prev)[0,0]
            
            cut.update({i:vol_sum - quad_prev - quad_new - 2*quad_prev_new})
            
            quad_prev = quad_prev + quad_new + 2*quad_prev_new
            temp_prev = temp_prev + temp_new
            A_temp_prev = A_temp_prev + A.dot(temp_new)
            
    cond = {}
    for i in cut:
        denominator = min(vol[i],vol_G - vol[i])
        cond.update({i:cut[i]/denominator})
    
    return cond, labels