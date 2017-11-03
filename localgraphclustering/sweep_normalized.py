import numpy as np

def sweep_normalized(p,g):
    """
    Computes a cluster using sweep cut and conductance as a criterion. 
    Each component of the input vector p is divided with the corresponding degree of the node. 

    Parameters 
    ----------

    p: numpy array
    A vector that is used to perform rounding.

    g: graph object   
                      
    Returns
    -------
           
    In a list of length 3 it returns the following.
           
    output 0: list
        Stores indices of the best clusters found by the last called rounding procedure.
           
    output 1: float
        Stores the value of the best conductance found by the last called rounding procedure.
                         
    output 2: list of objects
        A two dimensional list of objects. For example,
        sweep_profile[0] contains a numpy array with all conductances for all
        clusters that were calculated by the last called rounding procedure.
        sweep_profile[1] is a multidimensional list that contains the indices
        of all clusters that were calculated by the rounding procedure. For example,
        sweep_profile[1,5] is a list that contains the indices of the 5th cluster
        that was calculated by the rounding procedure. 
        The set of indices in sweep_profile[1][5] also correspond 
        to conductance in sweep_profile[0][5].
    """    
    n = g.adjacency_matrix.shape[0]

    nnz_idx = p.nonzero()[0]
    nnz_ct = nnz_idx.shape[0]

    sc_p = np.zeros(nnz_ct)
    for i in range(nnz_ct):
        degree = g.d[nnz_idx[i]]
        sc_p[i] = (-p[nnz_idx[i]]/degree)

    srt_idx = np.argsort(sc_p,axis=0)

    size_loop = nnz_ct
    if size_loop == n:
        size_loop = n - 1

    A_temp_prev = np.zeros((n,1))
    vol_sum = 0
    quad_prev = 0
    
    output = [[],[],[],[]]

    output[2] = [np.zeros(size_loop),[[] for jj in range(size_loop)]]
    output[3] = [[],[]]
        
    output[1] = 2
        
    for i in range(size_loop):

        idx = srt_idx[i]

        vol_sum = vol_sum + g.d[idx]

        quad_new = g.adjacency_matrix[idx,idx]
        quad_prev_new = A_temp_prev[idx,0]

        cut = vol_sum - quad_prev - quad_new - 2*quad_prev_new

        quad_prev = quad_prev + quad_new + 2*quad_prev_new
        A_temp_prev = A_temp_prev + g.adjacency_matrix[idx,:].T

        denominator = min(vol_sum,g.vol_G - vol_sum)

        cond = cut/denominator

        output[2][0][i] = cond
        current_support = (srt_idx[0:i+1]).tolist()
        output[2][1][i] = current_support

        if cond < output[1]:
            output[1] = cond
            output[0] = current_support
            
    return output

