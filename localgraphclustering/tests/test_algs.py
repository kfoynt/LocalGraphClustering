from localgraphclustering import *
import time
import numpy as np
import networkx as nx
import random

def load_example_graph(vtype,itype):
    return GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ",vtype=vtype,itype=itype)

def load_example_weighted_graph(vtype,itype):
    return GraphLocal("localgraphclustering/tests/data/neuro-fmri-01.edges",header=True,separator=" ",vtype=vtype,itype=itype)

def generate_random_3Dgraph(n_nodes, radius, seed=None):

    if seed is not None:
        random.seed(seed)
    
    # Generate a dict of positions
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}
    
    # Create random 3D network
    G = nx.random_geometric_graph(n_nodes, radius, pos=pos)

    return G

def test_GraphLocal_methods():
    g = load_example_graph(np.uint32,np.uint32)
    g1 = GraphLocal.from_shared(g.to_shared())
    assert(g == g1)
    
    g.largest_component()
    g.biconnected_components()
    g.core_number()
    ei,ej,e = [],[],[]
    for i in range(g._num_vertices):
        for j in range(g.ai[i],g.ai[i+1]):
            ei.append(i)
            ej.append(g.aj[j])
            e.append(g.adjacency_matrix.data[j])
    g1 = GraphLocal()
    g1.list_to_gl(ei,ej,e)

    assert(g1 == g)

    g1.discard_weights()

    # Compute triangle clusters and cluster metrics
    cond,cut,vol,cc,t = triangleclusters(g)
    minverts, minvals = g.local_extrema(cond,True)
    print("vertices with minimum conductance neighborhood:",minverts)

    # Test graph with more than one components
    G = GraphLocal("notebooks/datasets/neuro-fmri-01.edges",file_type = "edgelist", separator = " ", header = True)

    # Test drawing fuinctions
    # Read graph. This also supports gml and graphml format.
    # The data set contains pairwise similarities of blasted 
    # sequences of 232 proteins belonging to the amidohydrolase superfamily. 
    g = GraphLocal('notebooks/datasets/sfld_brown_et_al_amidohydrolases_protein_similarities_for_beh.graphml','graphml',' ')

    # Load pre-computed coordinates for nodes.
    pos = np.loadtxt('notebooks/datasets/sfld_brown_et_al_amidohydrolases_protein_similarities_for_beh.xy', dtype = 'float')

    groups = np.loadtxt('notebooks/datasets/sfld_brown_et_al_amidohydrolases_protein_similarities_for_beh.class', dtype = 'float')

    drawing = g.draw_groups(pos,groups,figsize=(15,15),nodesize_list=[10**2],edgealpha=0.05)

    # Find the solution of L1-regularized PageRank using localized accelerated gradient descent.
    # This method is the fastest among other l1-regularized solvers and other approximate PageRank solvers.
    reference_node = [218]
    l1_reg_vector = approximate_PageRank(g,reference_node,rho=1.0e-4,method="l1reg")

    # Call C++ version of sweep cut rounding on the l1-regularized PageRank solution.
    output_sc_fast = sweep_cut(g,l1_reg_vector)

    # Extract the partition for g and store it.
    l1_reg_vector_rounded = output_sc_fast[0]

    # Highlight local cluster
    drawing.highlight(l1_reg_vector_rounded,otheredges=True)
    drawing.nodesize(l1_reg_vector_rounded,10**2)
    drawing.nodecolor(l1_reg_vector_rounded,c='y')
    # Make reference node larger and thicker
    drawing.nodecolor(reference_node,facecolor='r',edgecolor='g',alpha=1)
    drawing.nodesize(reference_node,15**2)
    drawing.nodewidth(reference_node,3)
    drawing.show()

    #redraw the graph first
    drawing = g.draw_groups(pos,groups,figsize=(15,15),nodesize_list=[5**2],edgealpha=0.05)

    # Nodes circled whose color is not blue are missclassified
    drawing.nodecolor(l1_reg_vector_rounded,edgecolor='g',alpha=1)
    drawing.nodesize(l1_reg_vector_rounded,15**2)
    drawing.nodewidth(l1_reg_vector_rounded,5)
    drawing.show()

    N = generate_random_3Dgraph(n_nodes=200, radius=0.25, seed=1)
    pos = np.array(list(nx.get_node_attributes(N,'pos').values()))
    G = GraphLocal()
    G = G.from_networkx(N)
    drawing = G.draw(pos,edgealpha=0.01,nodealpha=0.5,
                      values=[random.uniform(0, 1) for i in range(200)])
    drawing = G.draw_groups(pos,[range(50),range(50,100),range(100,150)],edgealpha=0.01,nodealpha=0.5)


def test_sweepcut_self_loop():
    """ This is a regression test for sweep-cuts with self-loops """
    g = GraphLocal()
    # make a graph with a triangle that has a self-loop on one Vertex
    # connected to a lot of other vertices and where there are lots
    # of other things connected as well...
    #g.list_to_gl([0,0,1,2,2,2,2],[1,2,2,2])
    import networkx as nx
    G = nx.Graph()
    K6uK3 = nx.complete_graph(6)
    K6uK3.add_edge(5,6)
    K6uK3.add_edge(5,7)
    K6uK3.add_edge(6,7)
    K6uK3.add_edge(5,5)
    gloop = GraphLocal().from_networkx(K6uK3)
    S = [5,6,7]
    w = [3,2,1]
    Sp,phi1 = sweep_cut(gloop,(S,np.array(w)))
    assert(set(S) == set(Sp))
    assert(phi1 == 5/12)
    assert(phi1 == gloop.set_scores(S)["cond"])

def setup_fiedler_test(g):
    output_sp = fiedler(g)
    R = [1]
    R.extend(g.neighbors(R[0]))
    output_sp2 = fiedler_local(g,R)
    assert(np.all(output_sp2[0][1] >= -1.0e-6)) # make sure everything is almost positive.

def setup_fiedler_local_test(g):
    R = [1]
    R.extend(g.neighbors(R[0]))
    phi0 = g.set_scores(R)["cond"]
    sparse_vec = fiedler_local(g,R)[0]
    S = sweep_cut(g,sparse_vec)[0]
    phi1 = g.set_scores(S)["cond"]
    assert(phi1 <= phi0)

def setup_sweep_cut_test(g):
    tmp1 = sweep_cut(g,([1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]),cpp=False)
    tmp2 = sweep_cut(g,([1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]))
    assert(tmp1[1]==tmp2[1])

def setup_spectral_clustering_test(g):
    output_sp = fiedler(g)
    R = [1]
    R.extend(g.neighbors(R[0]))
    insize = len(R)
    output_sp2 = spectral_clustering(g,R,method="fiedler_local")[0]
    Z = set(R).union(output_sp2)
    assert(len(Z) == insize)
    phi0 = g.set_scores(R)["cond"]
    for method in ["fiedler_local","acl","l1reg","nibble"]:
        phi = g.set_scores(spectral_clustering(g, R, method=method)[0])["cond"]
        assert(phi <= phi0)

def setup_flow_clustering_test(g):
    # Call the global spectral partitioning algorithm.
    output_sp = fiedler(g)
    eig2 = output_sp

    # Round the eigenvector
    output_sc = sweep_cut(g,eig2)

    # Extract the partition for g and store it.
    R = output_sc[0]
    phi0 = g.set_scores(R)["cond"]

    for method in ["mqi","mqi_weighted","crd","sl"]:
        phi = g.set_scores(flow_clustering(g,R,method=method)[0])["cond"]
        assert(phi <= phi0)

def setup_flow_weighted_test():
    for vtype,itype in [(np.uint32,np.uint32),(np.uint32,np.int64),(np.int64,np.int64)]:
        g = load_example_graph(vtype,itype)
        g.discard_weights()
        cond1 = flow_clustering(g,range(20),method="mqi")[1]
        cond2 = flow_clustering(g,range(20),method="mqi_weighted")[1]
        # MQI_weighted should give the same result as MQI when running on unweighted graph
        assert(cond1 == cond2)
        cond1 = flow_clustering(g,range(20),method="sl")[1]
        cond2 = flow_clustering(g,range(20),method="sl_weighted")[1]
        # sl_weighted should give the same result as sl when running on unweighted graph
        assert(cond1 == cond2)
    for vtype,itype in [(np.uint32,np.uint32),(np.uint32,np.int64),(np.int64,np.int64)]:
        g = load_example_weighted_graph(vtype,itype)
        g1 = g.largest_component()
        cond1 = flow_clustering(g1,range(100),method="mqi_weighted")[1]
        cond2 = flow_clustering(g1,range(100),method="sl_weighted",delta=1.0e6)[1]
        # sl_weighted should give the same result as mqi_weighted when delta is large
        assert(cond1 == cond2)
    # create a 100 node clique
    edges = []
    for i in range(100):
        for j in range(i+1,100):
            # set edge weight of a five node subclique to be 10 and 1 elsewhere
            if i < 5 and j < 5:
                edges.append((i,j,10))
            else:
                edges.append((i,j,1))
    g = GraphLocal()
    g.list_to_gl(ei,ej,e)
    cluster = MQI_weighted(g,range(20))[0]
    # MQI_weighted should return the five node subclique with edge weight to be 10
    assert(np.array_equal(MQI_weighted(g,range(20))[0],np.array(range(5))))

def test_all_algs():
    for vtype,itype in [(np.uint32,np.uint32),(np.uint32,np.int64),(np.int64,np.int64)]:
        g = load_example_graph(vtype,itype)
        setup_fiedler_test(g)
        setup_fiedler_local_test(g)
        setup_sweep_cut_test(g)
        setup_spectral_clustering_test(g)
        setup_flow_clustering_test(g)
