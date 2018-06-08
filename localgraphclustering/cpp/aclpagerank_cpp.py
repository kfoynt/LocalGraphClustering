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
from .utility import determine_types
#from localgraphclustering.find_library import load_library


def aclpagerank_cpp(ai,aj,lib):

    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    
    #lib = load_library()

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.aclpagerank64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.aclpagerank32_64
    else:
        fun = lib.aclpagerank32

    #call C function
    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes.c_double,ctypes.c_double,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    return fun

def aclpagerank_run(fun,n,ai,aj,alpha,eps,seedids,maxsteps,xlength=10**6):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
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
