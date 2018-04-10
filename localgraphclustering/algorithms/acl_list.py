import time
import numpy as np

def acl_list(ref_node, g, alpha = 0.15, rho = 1.0e-5, max_iter = 100000, max_time = 100):
    
    n = g.adjacency_matrix.shape[0]
    
    r = np.zeros(n)
    p = np.zeros(n)
    
    nodes = []
    
    for i in ref_node:
        r[i] = 1
        thresh = rho*g.d[i]
        if r[i] > thresh:
            nodes.append(i)
    
    iter = 0        
    
    start = time.time()
         
    while len(nodes) > 0 and iter <= max_iter:
        
        idx = nodes[0]
        
        direction = r[idx]
        p[idx] = p[idx] + alpha*direction
        r[idx] = ((1-alpha)/2)*direction
        
        if r[idx] < rho*g.d[idx]:
            del nodes[0]
        else:
            nodes.append(idx) 
            del nodes[0]
        
        for u in range(g.adjacency_matrix.indptr[idx],g.adjacency_matrix.indptr[idx+1]):
            j = g.adjacency_matrix.indices[u]
            update = ((1-alpha)/2)*(direction/g.d[idx])*g.adjacency_matrix.data[u]
            r_new = r[j] + update
            thresh = rho*g.d[j]
            if r[j] <= thresh and r_new > thresh:
                nodes.append(j)  
            r[j] = r_new
            
        iter = iter + 1   
    
        end = time.time()
        
        if end - start > max_time:
            print("ACL: Maximum running time reached")
            break
            
    return p
