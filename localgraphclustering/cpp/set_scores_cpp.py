from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from .utility import determine_types, standard_types
from . import _graphlib


# Load the functions
def _setup_setscores_args(vtypestr, itypestr, fun, weighted):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,bool_type = standard_types(vtypestr,itypestr)

    fun.restype=None
    if weighted:
        fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(float_type, flags="C_CONTIGUOUS"),
                      ndpointer(float_type, flags="C_CONTIGUOUS"),
                      ctypes_vtype,
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ctypes_vtype,
                      ndpointer(float_type, flags="C_CONTIGUOUS"),
                      ndpointer(float_type, flags="C_CONTIGUOUS")]
    else:
        fun.argtypes=[ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ctypes_vtype,
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ctypes_vtype,
                      ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_itype, flags="C_CONTIGUOUS")]

    return fun


_graphlib_funs_set_scores64 = _setup_setscores_args(
    'int64','int64', _graphlib.set_scores64,False)
_graphlib_funs_set_scores32 = _setup_setscores_args(
    'uint32','uint32', _graphlib.set_scores32,False)
_graphlib_funs_set_scores32_64 = _setup_setscores_args(
    'uint32','int64', _graphlib.set_scores32_64,False)

_graphlib_funs_set_scores_weighted64 = _setup_setscores_args(
    'int64','int64', _graphlib.set_scores_weighted64,True)
_graphlib_funs_set_scores_weighted32 = _setup_setscores_args(
    'uint32','uint32', _graphlib.set_scores_weighted32,True)
_graphlib_funs_set_scores_weighted32_64 = _setup_setscores_args(
    'uint32','int64', _graphlib.set_scores_weighted32_64,True)


def _get_set_scores_cpp_types_fun(ai,aj,weighted):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_set_scores_weighted64 if weighted else _graphlib_funs_set_scores64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_set_scores_weighted32_64 if weighted else _graphlib_funs_set_scores32_64
    else:
        fun = _graphlib_funs_set_scores_weighted32 if weighted else _graphlib_funs_set_scores32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun

def set_scores_cpp(n,ai,aj,a,degrees,R,weighted):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_set_scores_cpp_types_fun(ai,aj,weighted)
    nR = len(R)
    R = np.array(R,dtype=vtype)
    if weighted:
        voltrue = np.zeros(1,dtype=np.double)
        cut = np.zeros(1,dtype=np.double)
        degrees = np.array(degrees,dtype=np.double)
        a = np.array(a,dtype=np.double)
    else:
        voltrue = np.zeros(1,dtype=itype)
        cut = np.zeros(1,dtype=itype)
    # flag is only for the use of julia, where list index starts from 1 instead of 0
    flag=0
    if weighted:
        fun(n,ai,aj,a,degrees,flag,R,nR,voltrue,cut)
    else:
        fun(n,ai,aj,flag,R,nR,voltrue,cut)

    return 1.0*voltrue[0],1.0*cut[0]