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
 *     y         - Initial solutions for l1-regularized PageRank algorithm.
 *                 If not provided then it is initialized to zero.
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
from .utility import determine_types, standard_types
from . import _graphlib


# Load the functions
def _setup_proxl1PRaccel_args(vtypestr, itypestr, fun):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = standard_types(vtypestr,itypestr)

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
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),ctypes_vtype,ctypes_vtype,
                  float_type]

    return fun


_graphlib_funs_proxl1PRaccel64 = _setup_proxl1PRaccel_args(
    'int64','int64', _graphlib.proxl1PRaccel64)
_graphlib_funs_proxl1PRaccel32 = _setup_proxl1PRaccel_args(
    'uint32','uint32', _graphlib.proxl1PRaccel32)
_graphlib_funs_proxl1PRaccel32_64 = _setup_proxl1PRaccel_args(
    'uint32','int64', _graphlib.proxl1PRaccel32_64)

"""
def proxl1PRaccel(ai,aj,lib):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)

    #lib = load_library()
    
    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.proxl1PRaccel64
    elif (vtype, itype) == (np.int32, np.int64):
        fun = lib.proxl1PRaccel32_64
    else:
        fun = lib.proxl1PRaccel32

    #call C function

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
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),ctypes_vtype,ctypes_vtype,
                  float_type]
    return fun
"""

def _get_proxl1PRaccel_cpp_types_fun(ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_proxl1PRaccel64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_proxl1PRaccel32_64
    else:
        fun = _graphlib_funs_proxl1PRaccel32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun

def proxl1PRaccel_cpp(ai,aj,a,ref_node,d,ds,dsinv,y=[],alpha = 0.15,rho = 1.0e-5,epsilon = 1.0e-4,maxiter = 10000,max_time = 100):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_proxl1PRaccel_cpp_types_fun(ai,aj)
    n = len(ai) - 1
    if type(ref_node) is not list:
        ref_node = np.array([ref_node],dtype = ctypes_vtype)
    else:
        ref_node = np.array(ref_node,dtype = ctypes_vtype)
    grad = np.zeros(n,dtype=float_type)
    p = np.zeros(n,dtype=float_type)

    if y == []:
        new_y = np.zeros(n,dtype=float_type)
    else:
        new_y = np.array(y,dtype=float_type)
    
    not_converged=fun(n,ai,aj,a,alpha,rho,ref_node,len(ref_node),d,ds,dsinv,epsilon,grad,p,new_y,maxiter,0,max_time)
    
    if y != []:
        for i in range(n):
            y[i] = new_y[i]

    return (not_converged,grad,p)






