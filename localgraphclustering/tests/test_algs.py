from localgraphclustering import *
import time

def test_algs():

    # Read graph. This also supports gml and graphml format.
    g = GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")

    # Call the global spectral partitioning algorithm.
    output_sp = fiedler(g)

    # Only one input graph is given, i.e., [g].
    # Extract the array from position 0 and store it.
    eig2 = output_sp

    # Round the eigenvector
    output_sc = sweep_cut(g,eig2)

    # Extract the partition for g and store it.
    eig2_rounded = output_sc[0]

    # Conductance before improvement
    print("Conductance before improvement:",g.compute_conductance(eig2_rounded))
    #print(eig2_rounded)

    # Start calling SimpleLocal
    start = time.time()
    output_SL = SimpleLocal(g,eig2_rounded)
    end = time.time()
    print("running time:",str(end-start)+"s")
    print(output_SL)

    # Conductance after improvement
    print("Conductance after improvement:",g.compute_conductance(output_SL[0]))

    # Compute triangle clusters and cluster metrics
    cond,cut,vol,cc,t = triangleclusters(g)
    minverts, minvals = g.local_extrema(cond,True)
    print("vertices with minimum conductance neighborhood:",minverts)

def test_fiedler():
    g = GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")
    output_sp = fiedler(g)
    output_sp2 = fiedler_local(g,[1,2,3,4,5,6,7,8,9,10])

def test_sweep_cut():
    g = GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")
    sweep_cut(g,([1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]))
