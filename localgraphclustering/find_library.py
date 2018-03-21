import sys
import imp
import ctypes
import _ctypes
import os

def find_path():
    if sys.platform == "linux2" or sys.platform == "linux":
        extension = ".so"
    elif sys.platform == "darwin":
        extension = ".dylib"
    elif sys.platform == "win32":
        extension = ".dll"
    else:
        print("Unknown system type!")
        return (True,0,0)
    cwd = os.getcwd()
    try:
        #try to load system-wide package first
        path = sys.path
        while ("" in path) or (cwd in path):
            if "" in path: path.remove("") 
            if cwd in path: path.remove(cwd)
        path_lgc = imp.find_module('localgraphclustering',path)[1]
    except ImportError:
        #if package is not installed, try to import from current directory
        path_lgc = imp.find_module('localgraphclustering')[1]
    return path_lgc+"/src/lib/graph_lib_test/libgraph"+extension

def load_library():
    #load library
    lib=ctypes.cdll.LoadLibrary(find_path())
    return lib

def reload_library(lib):
    handle = lib._handle
    name = lib._name
    del lib
    while is_loaded(name):
        _ctypes.dlclose(handle)
    return ctypes.cdll.LoadLibrary(name)

def is_loaded(lib):
    libp = os.path.abspath(lib)
    ret = os.system("lsof -p %d | grep %s > /dev/null" % (os.getpid(), libp))
    return (ret == 0)