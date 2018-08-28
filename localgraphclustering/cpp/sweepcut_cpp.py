# A python wrapper for sweep cut procedure
# A - the sparse matrix representing the symmetric graph
# ids - the order of vertices given
# results - the best set with the smallest conductance
# actual_length - the number of vertices in the best set
# num - the number of vertices given
# values - A vector scoring each vertex (e.g. pagerank value).
#          This will be sorted and turned into one of the other inputs.
# skip_sort - 0 for sweepcut_with_sorting and 1 for sweepcut_without_sorting
# degrees - user defined degrees, set it to be [] if not provided
# min_cond - minimum conductance

from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from .utility import determine_types
from . import _graphlib

# Load the functions
def _setup_sweepcut_args(vtypestr, itypestr, fun, sort=False):
    float_type = ctypes.c_double
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if vtypestr == 'int64' else (np.uint32, ctypes.c_uint32)
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if itypestr == 'int64' else (np.uint32, ctypes.c_uint32)

    fun.restype = ctypes_vtype

    if sort:
        fun.argtypes=[ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ctypes_vtype,ctypes_vtype,
                      ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes_vtype,
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_int]
    else:
        fun.argtypes=[ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ctypes_vtype,ctypes_vtype,
                      ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes_vtype,
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_int
                      ]

    return fun

_graphlib_funs_sweepcut_with_sorting64 = _setup_sweepcut_args(
    'int64','int64', _graphlib.sweepcut_with_sorting64, sort=True)
_graphlib_funs_sweepcut_without_sorting64 = _setup_sweepcut_args(
    'int64','int64', _graphlib.sweepcut_without_sorting64, sort=False)
_graphlib_funs_sweepcut_with_sorting32 = _setup_sweepcut_args(
    'uint32','uint32', _graphlib.sweepcut_with_sorting32, sort=True)
_graphlib_funs_sweepcut_without_sorting32 = _setup_sweepcut_args(
    'uint32','uint32', _graphlib.sweepcut_without_sorting32, sort=False)
_graphlib_funs_sweepcut_with_sorting32_64 = _setup_sweepcut_args(
    'uint32','int64', _graphlib.sweepcut_with_sorting32_64, sort=True)
_graphlib_funs_sweepcut_without_sorting32_64 = _setup_sweepcut_args(
    'uint32', 'int64', _graphlib.sweepcut_without_sorting32_64, sort=False)

#_graphlib_funs_sweepcut_with_sorting32 = _graphliblib.sweepcut_with_sorting32, sort=
#_graphlib_funs_sweepcut_without_sorting32 = _graphliblib.sweepcut_without_sorting32
#_graphlib_funs_sweepcut_with_sorting32_64 = _graphliblib.sweepcut_with_sorting32_64
#_graphlib_funs_sweepcut_without_sorting32_64 = _graphliblib.sweepcut_without_sorting32_64

def sweepcut_cpp(ai,aj,lib,flag):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)

    #lib = load_library()

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.sweepcut_with_sorting64 if flag == 0 else lib.sweepcut_without_sorting64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.sweepcut_with_sorting32_64 if flag == 0 else lib.sweepcut_without_sorting32_64
    else:
        fun = lib.sweepcut_with_sorting32 if flag == 0 else lib.sweepcut_without_sorting32

    #call C function
    fun.restype=ctypes_vtype

    if flag == 0:
        fun.argtypes=[ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ctypes_vtype,ctypes_vtype,
                      ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes_vtype,
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_int
                      #wrapped_ndptr(dtype=ctypes.c_double,ndim=1,flags="C_CONTIGUOUS")
                      ]
    else:
        fun.argtypes=[ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ctypes_vtype,ctypes_vtype,
                      ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes_vtype,
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                      ctypes.c_int
                      #wrapped_ndptr(dtype=ctypes.c_double,ndim=1,flags="C_CONTIGUOUS")
                      ]
    return fun

def sweepcut_run(fun,n,ai,aj,a,ids,num,values,flag,degrees = []):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)

    ids=np.array(ids,dtype=vtype)
    values=np.array(values,dtype=float_type)
    results=np.zeros(num,dtype=vtype)
    min_cond = np.array([0.0],dtype=float_type)
    degrees = np.array(degrees,dtype=float_type)

    if flag == 0:
        actual_length=fun(values,ids,results,num,n,ai,aj,a,0,min_cond,degrees,len(degrees)!=0)
    else:
        actual_length=fun(ids,results,num,n,ai,aj,a,0,min_cond,degrees,len(degrees)!=0)

    actual_results=np.empty(actual_length,dtype=vtype)
    actual_results[:]=[results[i] for i in range(actual_length)]
    min_cond = min_cond[0]
    return (actual_length,actual_results,min_cond)

def _get_sweepcut_cpp_types_fun(ai,aj,skip_sort):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_sweepcut_with_sorting64 if skip_sort == 0 else _graphlib_funs_sweepcut_without_sorting64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_sweepcut_with_sorting32_64 if skip_sort == 0 else _graphlib_funs_sweepcut_without_sorting32_64
    else:
        fun = _graphlib_funs_sweepcut_with_sorting32 if skip_sort == 0 else _graphlib_funs_sweepcut_without_sorting32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun

def sweepcut_simple(n,ai,aj,a,ids,num,values,skip_sort,degrees = []):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_sweepcut_cpp_types_fun(ai,aj,skip_sort)
    ids=np.array(ids,dtype=vtype)
    values=np.array(values,dtype=float_type)
    results=np.zeros(num,dtype=vtype)
    min_cond = np.array([0.0],dtype=float_type)
    degrees = np.array(degrees,dtype=float_type)

    if skip_sort == 0:
        actual_length=fun(values,ids,results,num,n,ai,aj,a,0,min_cond,degrees,len(degrees)!=0)
    else:
        actual_length=fun(ids,results,num,n,ai,aj,a,0,min_cond,degrees,len(degrees)!=0)

    actual_results=np.empty(actual_length,dtype=vtype)
    actual_results[:]=[results[i] for i in range(actual_length)]
    min_cond = min_cond[0]
    return (actual_length,actual_results,min_cond)
