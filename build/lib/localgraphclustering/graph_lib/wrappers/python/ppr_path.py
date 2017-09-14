# A python wrapper for ppr_path
# n - number of vertices
# ei,ej - edge list
# alpha - value of alpha
# eps - value of epsilon
# rho - value of rho
# seedids,nseedids - the set of indices for seeds
# xlength - the max number of ids in the solution vector
# xids, actual_length - the solution vector
# fun_id - 0 for aclpagerank64, 1 for aclpagerank32 and 2 for aclpagerank32_64

from operator import itemgetter
import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import *
from sys import platform

class ret_path_info:
    num_eps = 0
    epsilon = []
    conds = []
    cuts = []
    vols = []
    setsizes = []
    stepnums = []

class ret_rank_info:
    starts = []
    ends = []
    nodes = []
    deg_of_pushed = []
    size_of_solvec = []
    size_of_r = []
    val_of_push = []
    global_bcond = []
    nrank_changes = 0
    nrank_inserts = 0
    nsteps = 0
    size_for_best_cond = 0

def ppr_path(n,ai,aj,alpha,eps,rho,seedids,nseedids,xlength):
    
    float_type = c_double
        
    dt = np.dtype(ai[0])
    (itype, ctypes_itype) = (np.int64, c_int64) if dt.name == 'int64' else (np.uint32, c_uint32)
    dt = np.dtype(aj[0])
    (vtype, ctypes_vtype) = (np.int64, c_int64) if dt.name == 'int64' else (np.uint32, c_uint32)

    #load library
    if platform == "linux2":
        extension = ".so"
    elif platform == "darwin":
        extension = ".dylib"
    elif platform == "win32":
        extension = ".dll"
    else:
        print("Unknown system type!")
        return (True,0,0)
    lib=cdll.LoadLibrary("../../lib/graph_lib_test/./libgraph"+extension)

    if (vtype, itype) == (np.int64, np.int64):
        fun = lib.ppr_path64
    elif (vtype, itype) == (np.uint32, np.int64):
        fun = lib.ppr_path32_64
    else:
        fun = lib.ppr_path32

    #call C function
    maxstep = int(1.0/((1.0-alpha)*eps))
    class path_info(Structure):
        _fields_=[("num_eps",POINTER(c_int64)),
                  ("epsilon",POINTER(c_double * maxstep)),
                  ("conds",POINTER(c_double * maxstep)),
                  ("cuts",POINTER(c_double * maxstep)),
                  ("vols",POINTER(c_double * maxstep)),
                  ("setsizes",POINTER(c_int64 * maxstep)),
                  ("stepnums",POINTER(c_int64 * maxstep))]

    class rank_info(Structure):
        _fields_=[("starts",POINTER(c_int64 * maxstep)),
                  ("ends",POINTER(c_int64 * maxstep)),
                  ("nodes",POINTER(c_int64 * maxstep)),
                  ("deg_of_pushed",POINTER(c_int64 * maxstep)),
                  ("size_of_solvec",POINTER(c_int64 * maxstep)),
                  ("size_of_r",POINTER(c_int64 * maxstep)),
                  ("val_of_push",POINTER(c_double * maxstep)),
                  ("global_bcond",POINTER(c_double * maxstep)),
                  ("nrank_changes",POINTER(c_int64)),
                  ("nrank_inserts",POINTER(c_int64)),
                  ("nsteps",POINTER(c_int64)),
                  ("size_for_best_cond",POINTER(c_int64))]


    num_eps = c_int64(0)
    epsilon = (c_double * maxstep)()
    conds = (c_double * maxstep)()
    cuts  = (c_double * maxstep)()
    vols = (c_double * maxstep)()
    setsizes = (c_int64 * maxstep)()
    stepnums = (c_int64 * maxstep)()

    starts = (c_int64 * maxstep)()
    ends = (c_int64 * maxstep)()
    nodes = (c_int64 * maxstep)()
    deg_of_pushed = (c_int64 * maxstep)()
    size_of_solvec = (c_int64 * maxstep)()
    size_of_r = (c_int64 * maxstep)()
    val_of_push = (c_double * maxstep)()
    global_bcond = (c_double * maxstep)()
    nrank_changes = c_int64(0)
    nrank_inserts = c_int64(0)
    nsteps = c_int64(0)
    size_for_best_cond = c_int64(0)

    rank_stats = rank_info(pointer(starts),pointer(ends),pointer(nodes),
                       pointer(deg_of_pushed),pointer(size_of_solvec),
                       pointer(size_of_r),pointer(val_of_push),pointer(global_bcond),
                       pointer(nrank_changes),pointer(nrank_inserts),pointer(nsteps),
                       pointer(size_for_best_cond))
    
    eps_stats = path_info(pointer(num_eps),pointer(epsilon),pointer(conds),
                        pointer(cuts),pointer(vols),pointer(setsizes),pointer(stepnums))
    seedids = np.array(seedids,dtype=vtype)
    xids = np.zeros(xlength,dtype=vtype)
    fun.restype = ctypes_vtype
    fun.argtypes = [ctypes_vtype,ndpointer(ctypes_itype, flags="C_CONTIGUOUS"),
                   ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                   ctypes_vtype,c_double,c_double,c_double,
                   ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                   ctypes_vtype,
                   ndpointer(ctypes_vtype, flags="C_CONTIGUOUS"),
                   ctypes_vtype,path_info,rank_info]
    print("Calling C function")
    actual_length = fun(n,ai,aj,0,alpha,eps,rho,seedids,nseedids,xids,xlength,eps_stats,rank_stats)
    actual_xids = np.empty(actual_length,dtype=vtype)
    actual_xids[:] = [xids[i] for i in range(actual_length)]
    ret_eps_stats = ret_path_info()
    ret_eps_stats.num_eps = num_eps.value
    ret_eps_stats.epsilon = ret_eps_stats.epsilon + [epsilon[0:(num_eps.value+1)]]
    ret_eps_stats.conds = ret_eps_stats.conds + [conds[0:(num_eps.value+1)]]
    ret_eps_stats.cuts = ret_eps_stats.cuts + [cuts[0:(num_eps.value+1)]]
    ret_eps_stats.vols = ret_eps_stats.vols + [vols[0:(num_eps.value+1)]]
    ret_eps_stats.setsizes = ret_eps_stats.setsizes + [setsizes[0:(num_eps.value+1)]]
    ret_eps_stats.stepnums = ret_eps_stats.stepnums + [stepnums[0:(num_eps.value+1)]]

    ret_rank_stats = ret_rank_info()
    ret_rank_stats.nrank_changes = nrank_changes.value
    ret_rank_stats.nrank_inserts = nrank_inserts.value
    ret_rank_stats.nsteps = nsteps.value
    ret_rank_stats.size_for_best_cond = size_for_best_cond.value
    ret_rank_stats.starts = ret_rank_stats.starts + [starts[0:(nsteps.value+1)]]
    ret_rank_stats.ends = ret_rank_stats.ends + [ends[0:(nsteps.value+1)]]
    ret_rank_stats.nodes = ret_rank_stats.nodes + [nodes[0:(nsteps.value+1)]]
    ret_rank_stats.deg_of_pushed = ret_rank_stats.deg_of_pushed + [deg_of_pushed[0:(nsteps.value+1)]]
    ret_rank_stats.size_of_solvec = ret_rank_stats.size_of_solvec + [size_of_solvec[0:(nsteps.value+1)]]
    ret_rank_stats.size_of_r = ret_rank_stats.size_of_r + [size_of_r[0:(nsteps.value+1)]]
    ret_rank_stats.val_of_push = ret_rank_stats.val_of_push + [val_of_push[0:(nsteps.value+1)]]
    ret_rank_stats.global_bcond = ret_rank_stats.global_bcond + [global_bcond[0:(nsteps.value+1)]]




    return (actual_length,actual_xids,ret_eps_stats,ret_rank_stats)
