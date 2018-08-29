# A python wrapper for MQI
# ai,aj - graph in CSR
# n - number of nodes in the graph
# R - seed set
# nR - number of nodes in seed set
# actual_length - number of nodes in the optimal subset
# ret_set - optimal subset with the smallest conductance

from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from .utility import determine_types, standard_types
from . import _graphlib


def _setup_densest_subgraph_args(vtypestr, itypestr, fun):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = standard_types(vtypestr,itypestr)

    fun.restype=float_type
    fun.argtypes=[ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS")]

    return fun

_graphlib_funs_densest_subgraph64 = _setup_densest_subgraph_args(
    'int64','int64', _graphlib.densest_subgraph64)
_graphlib_funs_densest_subgraph32 = _setup_densest_subgraph_args(
    'uint32','uint32', _graphlib.densest_subgraph32)
_graphlib_funs_densest_subgraph32_64 = _setup_densest_subgraph_args(
    'uint32','int64', _graphlib.densest_subgraph32_64)

def _get_densest_subgraph_cpp_types_fun(ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_densest_subgraph64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_densest_subgraph32_64
    else:
        fun = _graphlib_funs_densest_subgraph32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun


"""
def densest_subgraph_cpp(ai,aj,lib):
    
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    
    #lib = load_library()

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.densest_subgraph64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.densest_subgraph32_64
    else:
        fun = lib.densest_subgraph32

    #call C function
    fun.restype=float_type
    fun.argtypes=[ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS")]
    return fun
"""

def densest_subgraph_cpp(n,ai,aj,a):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_densest_subgraph_cpp_types_fun(ai,aj)
    ret_set=np.zeros(n,dtype=vtype)
    actual_length=np.zeros(1,dtype=vtype)
    density=fun(n,ai,aj,a,0,ret_set,actual_length)
    actual_length=actual_length[0]
    actual_set=np.empty(actual_length,dtype=vtype)
    actual_set[:]=[ret_set[i] for i in range(actual_length)]
    
    return (density,actual_set)

