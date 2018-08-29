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
from .utility import determine_types, standard_types
from . import _graphlib


# Load the functions
def _setup_SimpleLocal_args(vtypestr, itypestr, fun):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = standard_types(vtypestr,itypestr)

    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  float_type]

    return fun


_graphlib_funs_SimpleLocal64 = _setup_SimpleLocal_args(
    'int64','int64', _graphlib.SimpleLocal64)
_graphlib_funs_SimpleLocal32 = _setup_SimpleLocal_args(
    'uint32','uint32', _graphlib.SimpleLocal32)
_graphlib_funs_SimpleLocal32_64 = _setup_SimpleLocal_args(
    'uint32','int64', _graphlib.SimpleLocal32_64)

"""
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
"""

def _get_SimpleLocal_cpp_types_fun(ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_SimpleLocal64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_SimpleLocal32_64
    else:
        fun = _graphlib_funs_SimpleLocal32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun

def SimpleLocal_cpp(n,ai,aj,nR,R,delta):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_SimpleLocal_cpp_types_fun(ai,aj)
    R=np.array(R,dtype=vtype)
    ret_set=np.zeros(n,dtype=vtype)
    actual_length=fun(n,nR,ai,aj,0,R,ret_set,delta)
    #print(actual_length)
    actual_set=np.empty(actual_length,dtype=vtype)
    actual_set[:]=[ret_set[i] for i in range(actual_length)]
    
    return (actual_length,actual_set)