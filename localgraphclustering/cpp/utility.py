import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes

def determine_types(ai,aj):
    float_type = ctypes.c_double
    dt = np.dtype(ai[0])
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)
    dt = np.dtype(aj[0])
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if dt.name == 'int64' else (np.uint32, ctypes.c_uint32)

    return float_type,vtype,itype,ctypes_vtype,ctypes_itype

def standard_types(vtypestr,itypestr):
    float_type = ctypes.c_double
    (vtype, ctypes_vtype) = (np.int64, ctypes.c_int64) if vtypestr == 'int64' else (np.uint32, ctypes.c_uint32)
    (itype, ctypes_itype) = (np.int64, ctypes.c_int64) if itypestr == 'int64' else (np.uint32, ctypes.c_uint32)
    return float_type,vtype,itype,ctypes_vtype,ctypes_itype
