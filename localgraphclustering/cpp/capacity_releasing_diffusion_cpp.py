from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from .utility import determine_types, standard_types
from . import _graphlib

# Load the functions
def _setup_crd_args(vtypestr, itypestr, fun):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = standard_types(vtypestr,itypestr)

    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes_vtype,ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype]

    return fun

_graphlib_funs_crd64 = _setup_crd_args(
  'int64','int64', _graphlib.capacity_releasing_diffusion64)
_graphlib_funs_crd32 = _setup_crd_args(
  'uint32','uint32', _graphlib.capacity_releasing_diffusion32)
_graphlib_funs_crd32_64 = _setup_crd_args(
  'uint32','int64', _graphlib.capacity_releasing_diffusion32_64)


def _get_crd_cpp_types_fun(ai,aj):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    if (vtype, itype) == (np.int64, np.int64):
        fun = _graphlib_funs_crd64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = _graphlib_funs_crd32_64
    else:
        fun = _graphlib_funs_crd32
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun

"""

def capacity_releasing_diffusion_cpp(ai,aj,lib):
    
    float_type,vtype,itype,ctypes_vtype,ctypes_itype = determine_types(ai,aj)
    
    #lib = load_library()

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.capacity_releasing_diffusion64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.capacity_releasing_diffusion32_64
    else:
        fun = lib.capacity_releasing_diffusion32

    #call C function
    fun.restype=ctypes_vtype
    fun.argtypes=[ctypes_vtype,
                  ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ndpointer(float_type, flags="C_CONTIGUOUS"),
                  ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype,ctypes_vtype,ctypes_vtype,ctypes_vtype,
                  ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                  ctypes_vtype]

    return fun
"""

def capacity_releasing_diffusion_cpp(n,ai,aj,a,U,h,w,iterations,ref_node):
    float_type,vtype,itype,ctypes_vtype,ctypes_itype,fun = _get_crd_cpp_types_fun(ai,aj)
    ref_node=np.array(ref_node,dtype=vtype)
    ret_set=np.zeros(n,dtype=vtype)
    actual_length=fun(n,ai,aj,a,0,ret_set,U,h,w,iterations,ref_node,len(ref_node))
    actual_set=np.empty(actual_length,dtype=vtype)
    actual_set[:]=[ret_set[i] for i in range(actual_length)]
    
    return actual_set