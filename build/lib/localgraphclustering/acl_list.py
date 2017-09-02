import time
import numpy as np

def acl_list(ref_node, g, alpha = 0.15, rho = 1.0e-5, max_iter = 100000, max_time = 100):
    
    n = g.A.shape[0]
    
    r = np.zeros(n)
    p = np.zeros(n)
    
    nodes = []
    
    nodes.append(ref_node)
    r[ref_node] = 1
    
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
        
        for u in range(g.A.indptr[idx],g.A.indptr[idx+1]):
            j = g.A.indices[u]
            update = ((1-alpha)/2)*(direction/g.d[idx])*g.A.data[u]
            r_new = r[j] + update
            thresh = rho*g.d[j]
            if r[j] < thresh and r_new >= thresh:
                nodes.append(j)  
            r[j] = r_new
            
        iter = iter + 1   
    
        end = time.time()
        
        if end - start > max_time:
            print("ACL: Maximum running time reached")
            break
            
    return p