# A python wrapper for aclpagerank
# n - number of vertices
# ai,aj - graph in CSR
# alpha - value of alpha
# eps - value of epsilon
# seedids,nseedids - the set of indices for seeds
# maxsteps - the max number of steps
# xlength - the max number of ids in the solution vector
# xids, actual_length - the solution vector
# values - the pagerank value vector for xids (already sorted in decreasing order)

from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from .utility import determine_types
from . import _graphlib

# Load the functions
def _setup_aclpagerank_weighted_args(vtypestr, itypestr, fun):
    float_type = ctypes.c_double
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if vtypestr == 'int64' else (np.uint32, ctypes.c_uint32)
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if itypestr == 'int64' else (np.uint32, ctypes.c_uint32)

    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes.c_double,ctypes.c_double,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    return fun

_graphlib_funs_aclpagerank_weighted64 = _setup_aclpagerank_weighted_args(
    'int64','int64', _graphlib.aclpagerank_weighted64)
_graphlib_funs_aclpagerank_weighted32 = _setup_aclpagerank_weighted_args(
    'uint32','uint32', _graphlib.aclpagerank_weighted32)
_graphlib_funs_aclpagerank_weighted32_64 = _setup_aclpagerank_weighted_args(
    'uint32','int64', _graphlib.aclpagerank_weighted32_64)

def _get_aclpagerank_weighted_cpp_types_fun(ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_aclpagerank_weighted64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_aclpagerank_weighted32_64
    else:
        fun = _graphlib_funs_aclpagerank_weighted32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun

"""
def aclpagerank_weighted_cpp(ai,aj,lib):

    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)

    #lib = load_library()
    
    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.aclpagerank_weighted64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.aclpagerank_weighted32_64
    else:
        fun = lib.aclpagerank_weighted32

    #call C function
    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes.c_double,ctypes.c_double,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    return fun
"""

def aclpagerank_weighted_cpp(n,ai,aj,a,alpha,eps,seedids,nseedids,maxsteps,xlength=10**6,flag=0):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_aclpagerank_weighted_cpp_types_fun(ai,aj)
    seedids=np.array(seedids,dtype=vtype)
    xids=np.zeros(xlength,dtype=vtype)
    values=np.zeros(xlength,dtype=float_type)
    a=np.array(a,dtype=float_type)
    actual_length=fun(n,ai,aj,a,flag,alpha,eps,seedids,nseedids,maxsteps,xids,xlength,values)
    if actual_length > xlength:
        xlength = actual_length
        xids=np.zeros(xlength,dtype=vtype)
        values=np.zeros(xlength,dtype=float_type)
        actual_length=fun(n,ai,aj,a,flag,alpha,eps,seedids,nseedids,maxsteps,xids,xlength,values)
    actual_values=values[0:actual_length]
    actual_xids=xids[0:actual_length]
    
    return (actual_length,actual_xids,actual_values)