import time
import numpy as np
from scipy import sparse as sp

def fista_dinput_dense(ref_node, g, alpha = 0.15, rho = 1.0e-5, epsilon = 1.0e-4, max_iter = 10000, max_time = 100):

    size_A = g.adjacency_matrix.shape
    n      = size_A[0]
    S      = ref_node
    S_and_neigh = S

    grad   = np.zeros(n)
    q      = np.zeros(n)
    y      = np.zeros(n)
    z      = np.zeros(n)

    d_sqrt_S = g.d_sqrt[S]
    dn_sqrt_S = g.dn_sqrt[S]

    if type(S) is list:
        l_S = len(S)
    else:
        l_S = 1

    grad[S] = -(alpha/l_S)*dn_sqrt_S

    grad_des = q[S] - grad[S]

    thres_vec = d_sqrt_S*(rho*alpha)

    S_filter = np.where(grad_des >= thres_vec)

    if S_filter[0].size == 0:
        print("Parameter rho is too large")
        return q

    scale_grad = np.multiply(grad[S],-dn_sqrt_S)

    max_sc_grad = scale_grad.max()

    iter = 1

    start = time.time()

    while (max_sc_grad > rho*alpha*(1+epsilon) and iter <= max_iter):

        z = y - grad

        nnz_z = z.nonzero()[0]
        idx_temp = np.where(z[nnz_z] >= rho*alpha*g.d_sqrt[nnz_z])[0]
        idx_pos = nnz_z[idx_temp]
        idx_temp = np.where(z[nnz_z] <= -rho*alpha*g.d_sqrt[nnz_z])[0]
        idx_neg = nnz_z[idx_temp]
        idx_pos_neg = np.append(idx_pos,idx_neg)
        idx_zero = diff(nnz_z,idx_pos_neg)

        q_old = q.copy()

        q[idx_pos] = z[idx_pos] - rho*alpha*g.d_sqrt[idx_pos]
        q[idx_neg] = z[idx_neg] + rho*alpha*g.d_sqrt[idx_neg]
        q[idx_zero] = 0

        if iter == 1:
            beta = 0
        else:
            beta = (1-np.sqrt(alpha))/(1+np.sqrt(alpha))

        y = q + beta*(q - q_old)

        grad = mat_vec_with_Q(g,alpha,y)
        grad[ref_node] = grad[ref_node] - (alpha/l_S)*g.dn_sqrt[ref_node]

        scale_grad = np.abs(np.multiply(grad,-g.dn_sqrt))

        max_sc_grad = scale_grad.max()

        iter = iter + 1

        end = time.time()

        if end - start > max_time:
            print("FISTA: Maximum running time reached")
            break

    q = np.abs(q)

    p = q.copy()
    p = np.multiply(q,g.d_sqrt)

    #print "Terminated at iteration %d." % iter

    return p

def mat_vec_with_Q(g,alpha,x):

    y = np.multiply(x,g.dn_sqrt)
    y = g.adjacency_matrix.dot(y)

    y = np.multiply(y,g.dn_sqrt)

    y = np.multiply(y,-(1-alpha)/2)
    y = y + np.multiply(x,(1+alpha)/2)

    return y

def diff(a, b):
    b = set(b)
    return np.asarray([aa for aa in a if aa not in b], dtype = 'int64')
