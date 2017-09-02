'''
 * INPUT:
 *     alpha     - teleportation parameter between 0 and 1
 *     rho       - l1-reg. parameter
 *     ref_node  - seed node
 *     ai,aj,a   - Compressed sparse row representation of A
 *     d         - vector of node strengths
 *     epsilon   - accuracy for termination criterion
 *     ds        - the square root of d
 *     dsinv     - 1/ds
 *     maxiter   - max number of iterations
 *
 * OUTPUT:
 *     p              - PageRank vector as a row vector
 *     not_converged  - flag indicating that maxiter has been reached
 *     grad           - last gradient
 *
'''

from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import platform
import imp

def proxl1PRaccel(ai,aj,a,ref_node,d,ds,dsinv,alpha = 0.15,rho = 1.0e-5,epsilon = 1.0e-4,maxiter = 10000,max_time = 100):
    n = len(ai) - 1
    if platform.architecture() == ('64bit', ''):
        float_type = np.float64
    else:
        float_type = np.float32

    dt = np.dtype(ai[0])
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)
    dt = np.dtype(aj[0])
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)

    #load library
    path_lgc = imp.find_module('localgraphclustering')[1]
    lib=ctypes.cdll.LoadLibrary(path_lgc+"/graph_lib/lib/graph_lib_test/libgraph.dylib")
    
    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.proxl1PRaccel64
    elif (vtype, itype) == (np.int32, np.int64):
        fun = lib.proxl1PRaccel32_64
    else:
        fun = lib.proxl1PRaccel32

    #call C function
    if type(ref_node) is not list:
        ref_node = np.array([ref_node],dtype = ctypes_vtype)
    else:
        ref_node = np.array(ref_node,dtype = ctypes_vtype)
    grad = np.zeros(n,dtype=float_type)
    p = np.zeros(n,dtype=float_type)
    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ctypes.c_double,ctypes.c_double,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),ctypes_vtype,
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes.c_double,
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),ctypes_vtype,ctypes_vtype,ctypes.c_double]
    not_converged=fun(n,ai,aj,a,alpha,rho,ref_node,len(ref_node),d,ds,dsinv,epsilon,grad,p,maxiter,0,max_time)


    return (not_converged,grad,p)





