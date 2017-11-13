def writeSMAT(g,name):
    n = g.adjacency_matrix.shape[0]
    ai = g.adjacency_matrix.indptr
    aj = g.adjacency_matrix.indices
    a = g.adjacency_matrix.data
    m = ai[n]
    wptr = open(name,"w")
    wptr.write("%d\t%d\t%d\n" % (n,n,m))
    for i in range(n):
        u = i
        for j in range(ai[i],ai[i+1]):
            v = aj[j]
            w = 1.0
            wptr.write("%d\t%d\t%f\n" % (u,v,w))
    wptr.close()