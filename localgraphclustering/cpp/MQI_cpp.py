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

# Load the functions
def _setup_MQI_args(vtypestr, itypestr, fun):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = standard_types(vtypestr,itypestr)

    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS")]

    return fun


_graphlib_funs_MQI64 = _setup_MQI_args(
    'int64','int64', _graphlib.MQI64)
_graphlib_funs_MQI32 = _setup_MQI_args(
    'uint32','uint32', _graphlib.MQI32)
_graphlib_funs_MQI32_64 = _setup_MQI_args(
    'uint32','int64', _graphlib.MQI32_64)


def _get_MQI_cpp_types_fun(ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_MQI64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_MQI32_64
    else:
        fun = _graphlib_funs_MQI32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun

"""
def MQI_cpp(ai,aj,lib):
    
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    
    #lib = load_library()

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.MQI64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.MQI32_64
    else:
        fun = lib.MQI32

    #call C function
    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS")]
    return fun
"""

def MQI_cpp(n,ai,aj,nR,R):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_MQI_cpp_types_fun(ai,aj)
    R=np.array(R,dtype=vtype)
    ret_set=np.zeros(nR,dtype=vtype)
    actual_length=fun(n,nR,ai,aj,0,R,ret_set)
    actual_set=np.empty(actual_length,dtype=vtype)
    actual_set[:]=[ret_set[i] for i in range(actual_length)]
    
    return (actual_length,actual_set)