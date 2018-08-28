from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from .utility import determine_types, standard_types
from . import _graphlib

# create functions
def _setup_triangleclusters_args(vtypestr, itypestr, fun):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = standard_types(vtypestr, itypestr)
    fun.restype = None
    fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
    return fun

_graphlib_funs_triangleclusters64 = _setup_triangleclusters_args(
    'int64','int64', _graphlib.triangleclusters64)
_graphlib_funs_triangleclusters32 = _setup_triangleclusters_args(
    'uint32','uint32', _graphlib.triangleclusters32)
_graphlib_funs_triangleclusters32_64 = _setup_triangleclusters_args(
    'uint32','int64', _graphlib.triangleclusters32_64)

def triangleclusters_cpp(n,ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_triangleclusters64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_triangleclusters32_64
    else:
        fun = _graphlib_funs_triangleclusters32

    cond = np.empty(n, dtype=float_type)
    cut = np.empty(n, dtype=float_type)
    vol = np.empty(n, dtype=float_type)
    cc = np.empty(n, dtype=float_type)
    t = np.empty(n, dtype=float_type)

    fun(n,ai,aj,cond,cut,vol,cc,t)

    return cond,cut,vol,cc,t
