import numpy as np
from operator import itemgetter
import platform
import ctypes

def list_to_CSR(filename):
    edge_tuples=[]

    #read data file
    print("read data file")
    f = open(filename)
    first_line = f.readline().strip()
    import re
    first_line = re.split(" ", first_line)
    n = long(first_line[0])
    m = long(first_line[2])
    ei = []
    ej = []
    w = []
    data = f.read()
    data = data.split()
    for i in range(m):
        ei += [int(data[3 * i])]
        ej += [int(data[3 * i + 1])]
        w += [data[3 * i + 2]]

    f.close()
    print("read data file, done!")

    #sort edge list
    print("sort edge list")
    edge_tuples=[]
    for i in range(0,m):
        edge_tuples=edge_tuples+[(ei[i],ej[i],w[i])]
    
    edge_tuples.sort(key=itemgetter(0,1))
    print("sort edge list, done!")

    ii32 = np.iinfo(np.uint32)
    itype = np.int64 if m > ii32.max else np.uint32
    vtype = np.int64 if n > ii32.max else np.uint32

    #convert edge list to CSR
    print("convert list to CSR")
    float_type = ctypes.c_double
    ai = np.zeros(n+1,dtype=itype)
    aj = np.zeros(m,dtype=vtype)
    a = np.zeros(m,dtype=float_type)
    i=0
    for item in edge_tuples:
        ai[item[0]+1]=ai[item[0]+1]+1
        aj[i]=item[1]
        a[i]=item[2]
        i=i+1
    for i in range(1,n+1):
        ai[i]=ai[i-1]+ai[i]
    print("convert list to CSR, done!")

    return (m,n,ai,aj,a)
