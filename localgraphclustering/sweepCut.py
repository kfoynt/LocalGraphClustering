"""
    DESCRIPTION
    -----------

    It implements rounding procedures for local graph clustering.

    Call help(SweepCut.__init__) to get the documentation for the variables of this class.
    Call help(SweepCut.name_of_function) to get the documentation for function name_of_function.

    CLASS VARIABLES
    ---------------

    1) best_cluster: list
                     Stores indices of the best clusters found by the last called rounding procedure.
    
    2) best_conductance: float, double
                         Stores the value of the best conductance found by the last called rounding procedure.
                         
    3) sweep_profile: list of objects
                      A two dimensional list of objects. For example,
                      sweep_profile[0] contains a numpy array with all conductances for all
                      clusters that were calculated by the last called rounding procedure.
                      sweep_profile[1] is a multidimensional list that contains the indices
                      of all clusters that were calculated by the rounding procedure. For example,
                      sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                      that was calculated by the rounding procedure. 
                      The set of indices in sweep_profile[1][5] also correspond 
                      to conductance in sweep_profile[0][5].
                      
    4) volume_profile: list of objects
                       A two dimensional list of objects. For example,
                       sweep_profile[0] contains a numpy array with all conductances for all
                       clusters that were calculated by the last called rounding procedure and also have 
                       volume less than or equal to the input parameter vol_user.
                       sweep_profile[1] is a multidimensional list that contains the indices
                       of all clusters that were calculated by the rounding procedure and also have 
                       volume less than or equal to the input parameter vol_user.                      

    FUNCTIONS 
    ---------

    1) sweep_cut_cpp(self, p, g, do_sort = 1, vol_user = 0)
    
    2) sweep_general(self, sc_p, g, vol_user = 0)
    
    3) sweep_normalized(self, p, g, vol_user = 0)
    
    4) sweep_unnormalized(self, p, g, vol_user = 0)
"""    

from scipy import sparse as sp
import numpy as np
from localgraphclustering.sweepcut_cpp import sweepcut_cpp

class sweepCut:
    
    def __init__(self):
        """
            CLASS VARIABLES
            ---------------

            1) best_cluster: list
                             Stores indices of the best clusters found by the last called rounding procedure.

            2) best_conductance: float
                                 Stores the value of the best conductance found by the last called rounding procedure.

            3) sweep_profile: list of objects
                              A two dimensional list of objects. For example,
                              sweep_profile[0] contains a numpy array with all conductances for all
                              clusters that were calculated by the last called rounding procedure.
                              sweep_profile[1] is a multidimensional list that contains the indices
                              of all clusters that were calculated by the rounding procedure. For example,
                              sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                              that was calculated by the rounding procedure. 
                              The set of indices in sweep_profile[1][5] also correspond 
                              to conductance in sweep_profile[0][5].

            4) sweep_profile: list of objects
                              A two dimensional list of objects. For example,
                              sweep_profile[0] contains a numpy array with all conductances for all
                              clusters that were calculated by the last called rounding procedure.
                              sweep_profile[1] is a multidimensional list that contains the indices
                              of all clusters that were calculated by the rounding procedure. For example,
                              sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                              that was calculated by the rounding procedure. 
                              The set of indices in sweep_profile[1][5] also correspond 
                              to conductance in sweep_profile[0][5].

            4) volume_profile: list of objects
                               A two dimensional list of objects. For example,
                               sweep_profile[0] contains a numpy array with all conductances for all
                               clusters that were calculated by the last called rounding procedure and also have 
                               volume less than or equal to the input parameter vol_user.
                               sweep_profile[1] is a multidimensional list that contains the indices
                               of all clusters that were calculated by the rounding procedure and also have 
                               volume less than or equal to the input parameter vol_user.                      
        """    
        self.best_cluster = []
        self.best_conductance = np.inf
        self.sweep_profile = []
        self.volume_profile = []
        
    def sweep_cut_cpp(self, p, g, do_sort = 1, vol_user = 0):
        """
            DESCRIPTION
            -----------

            Computes a cluster using sweep cut and conductance as a criterion. 
            It stores the results in class variable best_cluster.
            This method is a wrapper for the C++ rounding procedure. 

            Call help(SweepCut.__init__) to get the documentation for the variables of this class. 

            PARAMETERS 
            ----------

            p: numpy array
               A vector that is used to perform rounding.

            g: graph object   

            do_sort: binary 
                     default = 1 
                     If do_sort is equal to 1 then vector p is sorted in descending order first.
                     If do_sort is equal to 0 then vector p is not sorted.

            vol_user: float
                      Upper bound on volume for clusters added in variable volume_profile.
                      See documentation for variable volume_profile for details. 
                      
           RETURNS
           -------

           The output can be accessed from the sweepCut object that calls this function.
           
           best_cluster: float
                         Stores the value of the best conductance found by the last called rounding procedure.
        """        
        n = g.A.shape[0]
        
        uint_indptr = np.uint32(g.A.indptr) 
        uint_indices = np.uint32(g.A.indices)
        
        nnz_idx = p.nonzero()[0]
        nnz_ct = nnz_idx.shape[0]

        sc_p = np.zeros(nnz_ct)
        for i in range(nnz_ct):
            sc_p[i] = p[nnz_idx[i]]   
        
        (actual_length,bestclus,self.best_conductance) = sweepcut_cpp(n, uint_indptr, uint_indices, g.A.data, nnz_idx, nnz_ct, sc_p, 1 - do_sort)
        
        self.best_cluster = bestclus.tolist()    

    def sweep_general(self,p,g,vol_user = 0):
        """
            DESCRIPTION
            -----------

            Computes a cluster using sweep cut and conductance as a criterion. 
            It stores the results in class variables best_cluster, best_conductance, sweep_profile and volume_profile.

            Call help(SweepCut.__init__) to get the documentation for the variables of this class. 

            PARAMETERS 
            ----------

            p: numpy array
               A vector that is used to perform rounding.

            g: graph object   

            vol_user: float
                      Upper bound on volume for clusters added in variable volume_profile.
                      See documentation for variable volume_profile for details. 
                      
           RETURNS
           -------

           The output can be accessed from the sweepCut object that calls this function.
           
           best_cluster: list
                         Stores indices of the best clusters found by the last called rounding procedure.
           
           best_conductance: float
                             Stores the value of the best conductance found by the last called rounding procedure.
                         
           sweep_profile: list of objects
                           A two dimensional list of objects. For example,
                           sweep_profile[0] contains a numpy array with all conductances for all
                           clusters that were calculated by the last called rounding procedure.
                           sweep_profile[1] is a multidimensional list that contains the indices
                           of all clusters that were calculated by the rounding procedure. For example,
                           sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                           that was calculated by the rounding procedure. 
                           The set of indices in sweep_profile[1][5] also correspond 
                           to conductance in sweep_profile[0][5].
           
           volume_profile: list of objects
                            A two dimensional list of objects. For example,
                            sweep_profile[0] contains a numpy array with all conductances for all
                            clusters that were calculated by the last called rounding procedure and also have 
                            volume less than or equal to the input parameter vol_user.
                            sweep_profile[1] is a multidimensional list that contains the indices
                            of all clusters that were calculated by the rounding procedure and also have 
                            volume less than or equal to the input parameter vol_user. 
        """    
        n = g.A.shape[0]

        srt_idx = np.argsort(p,axis=0)
        
        size_loop = n - 1

        A_temp_prev = np.zeros((n,1))
        vol_sum = 0
        quad_prev = 0

        self.sweep_profile = [np.zeros(size_loop),[[] for jj in range(size_loop)]]
        self.volume_profile = [[],[]]

        for i in range(size_loop):

            idx = srt_idx[i]

            vol_sum = vol_sum + g.d[idx]

            quad_new = g.A[idx,idx]
            quad_prev_new = A_temp_prev[idx,0]

            cut = vol_sum - quad_prev - quad_new - 2*quad_prev_new

            quad_prev = quad_prev + quad_new + 2*quad_prev_new
            A_temp_prev = A_temp_prev + g.A[idx,:].T

            denominator = min(vol_sum,g.vol_G - vol_sum)

            cond = cut/denominator

            self.sweep_profile[0][i] = cond
            current_support = (srt_idx[0:i+1]).tolist()
            self.sweep_profile[1][i] = current_support

            if cond < self.best_conductance:
                self.best_conductance = cond
                self.best_cluster = current_support
                
            if vol_user < vol_sum and vol_sum < 2*g.vol_G/3:
                self.volume_profile[0].append(cond)
                self.volume_profile[1].append(current_support)

    def sweep_normalized(self,p,g,vol_user = 0):
        """
            DESCRIPTION
            -----------

            Computes a cluster using sweep cut and conductance as a criterion.
            Each component of the input vector p is divided with the corresponding degree of the node. 
            It stores the results in class variables best_cluster, best_conductance, sweep_profile and volume_profile.

            Call help(SweepCut.__init__) to get the documentation for the variables of this class. 

            PARAMETERS 
            ----------

            p: numpy array
               A vector that is used to perform rounding.

            g: graph object   

            vol_user: float
                      Upper bound on volume for clusters added in variable volume_profile.
                      See documentation for variable volume_profile for details. 
                      
           RETURNS
           -------

           The output can be accessed from the sweepCut object that calls this function.
           
           best_cluster: list
                         Stores indices of the best clusters found by the last called rounding procedure.
           
           best_conductance: float
                             Stores the value of the best conductance found by the last called rounding procedure.
                         
           sweep_profile: list of objects
                           A two dimensional list of objects. For example,
                           sweep_profile[0] contains a numpy array with all conductances for all
                           clusters that were calculated by the last called rounding procedure.
                           sweep_profile[1] is a multidimensional list that contains the indices
                           of all clusters that were calculated by the rounding procedure. For example,
                           sweep_profile[1,5] is a list that contains the indices of the 5th cluster
                           that was calculated by the rounding procedure. 
                           The set of indices in sweep_profile[1][5] also correspond 
                           to conductance in sweep_profile[0][5].
           
           volume_profile: list of objects
                            A two dimensional list of objects. For example,
                            sweep_profile[0] contains a numpy array with all conductances for all
                            clusters that were calculated by the last called rounding procedure and also have 
                            volume less than or equal to the input parameter vol_user.
                            sweep_profile[1] is a multidimensional list that contains the indices
                            of all clusters that were calculated by the rounding procedure and also have 
                            volume less than or equal to the input parameter vol_user. 
        """  
        [n,n] = g.A.shape

        nnz_idx = p.nonzero()[0]
        nnz_ct = nnz_idx.shape[0]

        sc_p = np.zeros((nnz_ct, 1))
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

        self.sweep_profile = [np.zeros(size_loop),[[] for jj in range(size_loop)]]
        self.volume_profile = [[],[]]
        
        for i in range(size_loop):

            idx = nnz_idx[srt_idx[i]]

            vol_sum = vol_sum + g.d[idx]

            quad_new = g.A[idx,idx]
            quad_prev_new = A_temp_prev[idx,0]

            cut = vol_sum - quad_prev - quad_new - 2*quad_prev_new

            quad_prev = quad_prev + quad_new + 2*quad_prev_new
            A_temp_prev = A_temp_prev + g.A[idx,:].T

            denominator = min(vol_sum,g.vol_G - vol_sum)

            cond = cut/denominator

            self.sweep_profile[0][i] = cond
            current_support = (nnz_idx[srt_idx[0:i+1]])[:,0].tolist()
            self.sweep_profile[1][i] = current_support

            if cond < self.best_conductance:
                self.best_conductance = cond
                self.best_cluster = current_support
                
            if vol_user < vol_sum and vol_sum < 2*g.vol_G/3:
                self.volume_profile[0].append(cond)
                self.volume_profile[1].append(current_support)
                
#    def sweep_unnormalized(self,p,g,vol_user = 0):
#
#        [n,n] = g.A.shape
#
#        nnz_idx = p.nonzero()[0]
#        nnz_ct = nnz_idx.shape[0]
#
#        sc_p = np.zeros((nnz_ct, 1))
#        for i in range(nnz_ct):
#            degree = g.d[nnz_idx[i]]
#            sc_p[i] = (-p[nnz_idx[i]])
#
#        srt_idx = np.argsort(sc_p,axis=0)
#        
#        size_loop = nnz_ct
#        if size_loop == n:
#            size_loop = n - 1
#
#        A_temp_prev = np.zeros((n,1))
#        vol_sum = 0
#        quad_prev = 0
#
#        self.sweep_profile = [np.zeros(size_loop),[[] for jj in range(size_loop)]]
#        self.volume_profile = [[],[]]
#
#        for i in range(size_loop):
#
#            idx = nnz_idx[srt_idx[i]]
#
#            vol_sum = vol_sum + g.d[idx]
#
#            quad_new = g.A[idx,idx]
#            quad_prev_new = A_temp_prev[idx,0]
#
#            cut = vol_sum - quad_prev - quad_new - 2*quad_prev_new
#
#            quad_prev = quad_prev + quad_new + 2*quad_prev_new
#            A_temp_prev = A_temp_prev + g.A[idx,:].T
#
#            denominator = min(vol_sum,g.vol_G - vol_sum)
#
#            cond = cut/denominator
#
#            self.sweep_profile[0][i] = cond
#            current_support = (nnz_idx[srt_idx[0:i+1]])[:,0].tolist()
#            self.sweep_profile[1][i] = current_support
#
#            if cond < self.best_conductance:
#                self.best_conductance = cond[0,0]
#                self.best_cluster = current_support
#                
#            if vol_user < vol_sum and vol_sum < 2*g.vol_G/3:
#                self.volume_profile[0].append(cond[0,0])
#                self.volume_profile[1].append(current_support)

# def sweep_cut_degree_sqrt_normalized_map(A,d,p,ref_nodes):   
#     
#     ref_nodes = np.asarray(ref_nodes)
#         
#     [n,n] = A.shape
#     
#     nnz_ct = p.count_nonzero()
#     nnz_idx = p.nonzero()[0]
#     
#     sc_p = np.zeros((nnz_ct, 1))
#     for i in range(nnz_ct):
#         degree = A[nnz_idx[i],:].sum(axis=1)[0,0]
#         sc_p[i] = (-p[nnz_idx[i]]/np.sqrt(d[ref_nodes[nnz_idx[i]]]))[0,0]
#             
#     srt_idx = np.argsort(sc_p,axis=0)
#     
#     best_conductance = np.inf
#     best_cluster = []
#     
#     size_loop = nnz_ct
#     if size_loop == n:
#         size_loop = n - 1
#         
#     A_temp_prev = sp.csr_matrix((n,1),dtype=int)
#     vol_sum = 0
#     quad_prev = 0
#         
#     for i in range(size_loop):
#         
#         idx = ref_nodes[nnz_idx[srt_idx[i]]]
#     
#         vol_sum = vol_sum + A[idx,:].sum(axis=1)[0,0]
#             
#         quad_new = A[idx,idx]
#         quad_prev_new = A_temp_prev[idx,0]
#            
#         cut = vol_sum - quad_prev - quad_new - 2*quad_prev_new
#             
#         quad_prev = quad_prev + quad_new + 2*quad_prev_new
#         A_temp_prev = A_temp_prev + A[idx,:].T
#         
#         cond = cut/vol_sum
#         
#         if cond < best_conductance:
#             best_conductance = cond
#             best_cluster = (ref_nodes[nnz_idx[srt_idx[0:i+1]]])[:,0].tolist()
#             
#     return best_cluster
