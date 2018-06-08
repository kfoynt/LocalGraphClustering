# A python wrapper for SimpleLocal
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
from .utility import determine_types
#from localgraphclustering.find_library import load_library


def SimpleLocal_cpp(ai,aj,lib):
    
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    
    #lib = load_library()

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.SimpleLocal64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.SimpleLocal32_64
    else:
        fun = lib.SimpleLocal32

    #call C function
    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  float_type]
    return fun

def SimpleLocal_run(fun,n,ai,aj,nR,R,delta):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    R=np.array(R,dtype=vtype)
    ret_set=np.zeros(n,dtype=vtype)
    actual_length=fun(n,nR,ai,aj,0,R,ret_set,delta)
    #print(actual_length)
    actual_set=np.empty(actual_length,dtype=vtype)
    actual_set[:]=[ret_set[i] for i in range(actual_length)]
    
    return (actual_length,actual_set)