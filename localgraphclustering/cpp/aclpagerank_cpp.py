# A python wrapper for aclpagerank
# n - number of vertices
# ai,aj - graph in CSR
# alpha - value of alpha
# eps - value of epsilon
# seedids - the set of indices for seeds
# maxsteps - the max number of steps
# xlength - the max number of ids in the solution vector
# xids, actual_length - the solution vector
# values - the pagerank value vector for xids (already sorted in decreasing order)

from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from .utility import determine_types, standard_types
from . import _graphlib


# Load the functions
def _setup_aclpagerank_args(vtypestr, itypestr, fun):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = standard_types(vtypestr,itypestr)

    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes.c_double,ctypes.c_double,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    return fun


_graphlib_funs_aclpagerank64 = _setup_aclpagerank_args(
    'int64','int64', _graphlib.aclpagerank64)
_graphlib_funs_aclpagerank32 = _setup_aclpagerank_args(
    'uint32','uint32', _graphlib.aclpagerank32)
_graphlib_funs_aclpagerank32_64 = _setup_aclpagerank_args(
    'uint32','int64', _graphlib.aclpagerank32_64)


def _get_aclpagerank_cpp_types_fun(ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_aclpagerank64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_aclpagerank32_64
    else:
        fun = _graphlib_funs_aclpagerank32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun

def aclpagerank_cpp(n,ai,aj,alpha,eps,seedids,maxsteps,xlength=10**6):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_aclpagerank_cpp_types_fun(ai,aj)
    nseedids = len(seedids)
    seedids = np.array(seedids,dtype=vtype)
    xids = np.zeros(xlength,dtype=vtype)
    values = np.zeros(xlength,dtype=float_type)
    # flag is only for the use of julia, where list index starts from 1 instead of 0
    flag=0
    actual_length=fun(n,ai,aj,flag,alpha,eps,seedids,nseedids,maxsteps,xids,xlength,values)
    if actual_length > xlength:
        xlength = actual_length
        xids=np.zeros(xlength,dtype=vtype)
        values=np.zeros(xlength,dtype=float_type)
        actual_length=fun(n,ai,aj,flag,alpha,eps,seedids,nseedids,maxsteps,xids,xlength,values)
    actual_values=values[0:actual_length]
    actual_xids=xids[0:actual_length]

    return (actual_length,actual_xids,actual_values)
