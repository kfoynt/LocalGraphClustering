# A python wrapper for MQI
# ei,ej - edge list representation of graph
# n - number of nodes in the graph
# R - seed set
# nR - number of nodes in seed set
# actual_length - number of nodes in the optimal subset
# ret_set - optimal subset with the smallest conductance

from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from sys import platform

def MQI(n,ai,aj,nR,R):
    
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
