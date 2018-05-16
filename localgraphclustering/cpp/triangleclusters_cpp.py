from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

def triangleclusters_cpp(n,ai,aj,lib):

    float_type = ctypes.c_double

    dt = np.dtype(ai[0])
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)
    dt = np.dtype(aj[0])
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.triangleclusters64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.triangleclusters32_64
    else:
        fun = lib.triangleclusters32

    cond = np.empty(n, dtype=float_type)
    cut = np.empty(n, dtype=float_type)
    vol = np.empty(n, dtype=float_type)
    cc = np.empty(n, dtype=float_type)
    t = np.empty(n, dtype=float_type)

    fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

    fun(n,ai,aj,cond,cut,vol,cc,t)

    return cond,cut,vol,cc,t