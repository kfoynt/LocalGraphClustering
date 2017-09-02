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
from sys import platform

def proxl1PRaccel(ai,aj,a,ref_node,d,ds,dsinv,alpha = 0.15,rho = 1.0e-5,epsilon = 1.0e-4,maxiter = 10000,max_time = 100):
    n = len(ai) - 1
    float_type = ctypes.c_double
    dt = np.dtype(ai[0])
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)
    dt = np.dtype(aj[0])
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)

    #load library
    if platform == "linux2":
        extension = ".so"
    elif platform == "darwin":
        extension = ".dylib"
    elif platform == "win32":
        extension = ".dll"
    else:
        print("Unknown system type!")
        return (True,0,0)
    lib=ctypes.cdll.LoadLibrary("../../lib/graph_lib_test/./libgraph"+extension)
    
    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.proxl1PRaccel64
    elif (vtype, itype) == (np.uint32, np.int64):
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
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  float_type,float_type,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),ctypes_vtype,
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),float_type,
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),ctypes_vtype,ctypes_vtype,
                  float_type]
    not_converged=fun(n,ai,aj,a,alpha,rho,ref_node,len(ref_node),d,ds,dsinv,epsilon,grad,p,maxiter,0,max_time)


    return (not_converged,grad,p)





