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
#from localgraphclustering.find_library import load_library


def MQI_cpp(n,ai,aj,nR,R,lib):
    
    float_type = ctypes.c_double
    
    dt = np.dtype(ai[0])
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)
    dt = np.dtype(aj[0])
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)
    
    #lib = load_library()

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.MQI64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.MQI32_64
    else:
        fun = lib.MQI32

    #call C function
    R=np.array(R,dtype=vtype)
    ret_set=np.zeros(nR,dtype=vtype)
    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS")]
    actual_length=fun(n,nR,ai,aj,0,R,ret_set)
    actual_set=np.empty(actual_length,dtype=vtype)
    actual_set[:]=[ret_set[i] for i in range(actual_length)]
    
    return (actual_length,actual_set)
