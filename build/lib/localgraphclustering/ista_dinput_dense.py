import time
import numpy as np
from scipy import sparse as sp

def ista_dinput_dense(ref_node, g, alpha = 0.15, rho = 1.0e-5, epsilon = 1.0e-2, max_iter = 10000, max_time = 100):
    
    size_A = g.A.shape
    n      = size_A[0]
    S      = ref_node
    S_and_neigh = S
    
    grad   = np.zeros(n, dtype = np.float64)
    q      = np.zeros(n, dtype = np.float64)

    d_sqrt_S = g.d_sqrt[S]
    dn_sqrt_S = g.dn_sqrt[S]
    
    grad[S] = -alpha*dn_sqrt_S
    
    grad_des = q[S] - grad[S]
       
    thres_vec = d_sqrt_S*(rho*alpha)
    
    S_filter = np.where(grad_des >= thres_vec)
    
    if S_filter[0].size == 0:
        print("Parameter rho is too large")
        return []
    
    scale_grad = np.multiply(grad[S],-dn_sqrt_S)
        
    max_sc_grad = scale_grad.max()
        
    iter = 1    
    
    start = time.time()
    
    while (max_sc_grad > rho*alpha*(1+epsilon) and iter <= max_iter):
        
        z = q - grad
        
        idx_pos = np.where(z >= rho*alpha*g.d_sqrt)[0]
        
        q[idx_pos] = z[idx_pos] - rho*alpha*g.d_sqrt[idx_pos]
        
        grad = mat_vec_with_Q(g,alpha,q)
        grad[ref_node] = grad[ref_node] - alpha*g.dn_sqrt[ref_node]
        
        scale_grad = np.multiply(grad,-g.dn_sqrt)
        scale_grad.data = np.abs(scale_grad.data)

        max_sc_grad = scale_grad.max()
        
        iter = iter + 1  
        
        end = time.time()
        
        if end - start > max_time:
            print("ISTA: Maximum running time reached")
            break
            
    p = np.multiply(q,g.d_sqrt)
        
    #print "Terminated at iteration %d." % iter
    
    return p

def mat_vec_with_Q(g,alpha,x):
    
    y = np.multiply(x,g.dn_sqrt)
    
    y = g.A.dot(y)
    
    y = np.multiply(y,g.dn_sqrt)

    y = np.multiply(y,-(1-alpha)/2)
    y = y + np.multiply(x,(1+alpha)/2)
    
    return y
    
