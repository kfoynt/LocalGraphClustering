from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
#from localgraphclustering.find_library import load_library


def capacity_releasing_diffusion_cpp(n,ai,aj,a,U,h,w,iterations,ref_node,lib):
    
    float_type = ctypes.c_double
    
    dt = np.dtype(ai[0])
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)
    dt = np.dtype(aj[0])
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)
    
    #lib = load_library()

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.capacity_releasing_diffusion64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.capacity_releasing_diffusion32_64
    else:
        fun = lib.capacity_releasing_diffusion32

    #call C function
    ref_node=np.array(ref_node,dtype=vtype)
    ret_set=np.zeros(n,dtype=vtype)
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
    actual_length=fun(n,ai,aj,a,0,ret_set,U,h,w,iterations,ref_node,len(ref_node))
    actual_set=np.empty(actual_length,dtype=vtype)
    actual_set[:]=[ret_set[i] for i in range(actual_length)]
    
    return actual_set