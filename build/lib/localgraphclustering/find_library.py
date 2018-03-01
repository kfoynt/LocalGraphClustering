from sys import platform
import imp
import ctypes

def load_library():
    #load library
    if platform == "linux2" or platform == "linux":
        extension = ".so"
    elif platform == "darwin":
        extension = ".dylib"
    elif platform == "win32":
        extension = ".dll"
    else:
        print("Unknown system type!")
        return (True,0,0)
    path_lgc = imp.find_module('localgraphclustering')[1]
    lib=ctypes.cdll.LoadLibrary(path_lgc+"/src/lib/graph_lib_test/libgraph"+extension)
    return lib