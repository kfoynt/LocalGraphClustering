import time
import numpy as np
from scipy import sparse as sp
from localgraphclustering.fista_dinput_dense import fista_dinput_dense
from localgraphclustering.proxl1PRaccel import proxl1PRaccel

def multiclass_label_prediction(labels, g, alpha = 0.15, rho = 1.0e-10, epsilon = 1.0e-2, max_iter = 10000, max_time = 100, cpp = True):
    """
       DESCRIPTION
       -----------
       
       This function predicts labels for unlabelled nodes. For details refer to:
       D. Gleich and M. Mahoney. Variational 
       Using Local Spectral Methods to Robustify Graph-Based Learning Algorithms. SIGKDD 2015.
       https://www.stat.berkeley.edu/~mmahoney/pubs/robustifying-kdd15.pdf
       
       PARAMETERS (mandatory)
       ----------------------
       
       labels:  list of lists
                Each list of this list corresponds to indices of nodes that are assumed to belong in
                a certain class. For example, list[i] is a list of indices of nodes that are assumed to 
                belong in class i.
                  
       g:         graph object
       
       PARAMETERS (optional)
       ---------------------
      
       alpha: float, double
              default == 0.15
              Teleportation parameter of the personalized PageRank linear system.
              The smaller the more global the personalized PageRank vector is.
          
       rho:   float, double
              defaul == 1.0e-10
              Regularization parameter for the l1-norm of the model.
              
       For details of these parameters please refer to: K. Fountoulakis, F. Roosta-Khorasani, 
       J. Shun, X. Cheng and M. Mahoney. Variational Perspective on Local Graph Clustering. arXiv:1602.01886, 2017
       arXiv link:https://arxiv.org/abs/1602.01886 
          
       epsilon: float, double
                default == 1.0e-2
                Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.
          
       max_iter: integer
                 default = 10000
                 Maximum number of iterations of FISTA.
                 
       max_time: float, double
                 default = 100
                 Maximum time in seconds
                 
       cpp: boolean
            default = True
            Use the faster C++ version of FISTA or not.
       
       RETURNS
       ------
       
       classes: list of indices that holds the class for each node.
                For example classes[i] is the class of node i.
       
       ranks: list of lists. Each componenent of the list is a list that holds the rank
              of the nodes for each class. For details see [1].
       
       diffusions: a list of numpy arrays. Each array in this list corresponds to the diffusion vector
                   returned by personalized PageRank for each rank. For details see [1].
                   
       [1] D. Gleich and M. Mahoney. Variational 
       Using Local Spectral Methods to Robustify Graph-Based Learning Algorithms. SIGKDD 2015.
       https://www.stat.berkeley.edu/~mmahoney/pubs/robustifying-kdd15.pdf
    """   
   
    n = g.A.shape[0]

    diffusions = [] 
    ranks = []
    
    classes = []
    
    for labels_i in labels: 
        
        if not cpp:
    
            output = fista_dinput_dense(labels_i, g, alpha = alpha, rho = rho, epsilon = epsilon, max_iter = max_iter, max_time = max_time)
        else: 
            uint_indptr = np.uint32(g.A.indptr) 
            uint_indices = np.uint32(g.A.indices)
        
            (not_converged,grad,output) = proxl1PRaccel(uint_indptr, uint_indices, g.A.data, labels_i, g.d, g.d_sqrt, g.dn_sqrt, alpha = alpha, rho = rho, epsilon = epsilon, maxiter = max_iter, max_time = max_time)
        
        p = np.zeros(n)
        for i in range(n):
            p[i] = output[i]
        
        diffusions.append(p)
        
        index = (-p).argsort(axis=0)
        rank = np.empty(n, int)
        rank[index] = np.arange(n)
        
        ranks.append(rank)
        
    l_labels = len(labels)
    
    for i in range(n):
        min_rank = n+1
        class_ = l_labels + 1
        for j in range(l_labels):
            rank = ranks[j][i]
            if rank < min_rank:
                min_rank = rank
                class_ = j
        classes.append(class_)
        
    return classes, ranks, diffusions
