from localgraphclustering import *
import time

def test_algs():

    # Read graph. This also supports gml and graphml format.
    g = graph_class_local.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")

    # Create an object for global spectral partitioning
    sp = spectral_partitioning.Spectral_partitioning()

    # Call the global spectral partitioning algorithm.
    output_sp = sp.produce([g])

    # Only one input graph is given, i.e., [g]. 
    # Extract the array from position 0 and store it.
    eig2 = output_sp[0]

    # Create an object for the sweep cut rounding procedure.
    sc = sweepCut_general.SweepCut_general()

    # Round the eigenvector
    output_sc = sc.produce([g],p=eig2)

    # Extract the partition for g and store it.
    eig2_rounded = output_sc[0][0]

    # Create an object for subgraph node partitioning.
    SL_fast = SimpleLocal_fast.SimpleLocal_fast()

    # Conductance before improvement
    print("Conductance before improvement:",g.compute_conductance(eig2_rounded))
    #print(eig2_rounded)

    # Start calling SimpleLocal
    start = time.time()
    output_SL_fast = SL_fast.produce([g],[eig2_rounded])
    end = time.time()
    print("running time:",str(end-start)+"s")
    print(output_SL_fast)

    # Conductance after improvement
    print("Conductance after improvement:",g.compute_conductance(output_SL_fast[0][0]))

    output_SL = output_SL_fast[0][0]

