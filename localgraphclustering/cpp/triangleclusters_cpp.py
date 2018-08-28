from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from .utility import determine_types

def triangleclusters_cpp(ai,aj,lib):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.triangleclusters64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.triangleclusters32_64
    else:
        fun = lib.triangleclusters32
    fun.restype = None
    fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    return fun

def triangleclusters_run(fun,n,ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    cond = np.empty(n, dtype=float_type)
    cut = np.empty(n, dtype=float_type)
    vol = np.empty(n, dtype=float_type)
    cc = np.empty(n, dtype=float_type)
    t = np.empty(n, dtype=float_type)

    fun(n,ai,aj,cond,cut,vol,cc,t)

    return cond,cut,vol,cc,t
