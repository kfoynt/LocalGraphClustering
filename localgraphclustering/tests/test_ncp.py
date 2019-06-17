import localgraphclustering as lgc
import pytest
#from functools import partial

def load_example_graph():
    return lgc.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")

def test_ncp():
    G = load_example_graph()
    ncp = lgc.NCPData(G)
    df = ncp.as_data_frame()
    assert len(df) == 0
    ncp.mqi(nthreads=1,ratio=1.0)
    df = ncp.as_data_frame()
    assert len(df) == G._num_vertices
    #func = lambda G,R: lgc.flow_clustering(G,R,method="mqi")[0]
    func = lgc.partialfunc(lgc.flow_clustering, method="mqi")
    ncp = lgc.NCPData(G)
    ncp.add_set_samples([[1]],nthreads=1,method=func,methodname="mqi")
    ncp.add_random_neighborhood_samples(ratio=2,nthreads=1,method=func,methodname="mqi")


def test_ncp_mqi():
    G = load_example_graph()
    df = lgc.NCPData(G).mqi(ratio=1)

def test_ncp_one_thread():
    G = load_example_graph()
    df = lgc.NCPData(G).mqi(ratio=2,nthreads=1)

def _second(G,R):
    return R, []
# this used to always catch some errors...
def test_custom_ncp():
    G = load_example_graph()
    ncp = lgc.NCPData(G)
    ncp.add_random_neighborhood_samples(ratio=1.0,
        method=_second, methodname="neighborhoods", nthreads=16)

def test_ncp_read_write():
    G = load_example_graph()
    ncp = lgc.NCPData(G).approxPageRank(ratio=2)
    R1 = ncp.input_set(0)
    S1 = ncp.output_set(0)
    R2 = ncp.input_set(1)
    S2 = ncp.output_set(1)
    ncp.write("myncp")
    ncp2 = lgc.NCPData.from_file("myncp.pickle", G)
    assert(R1 == ncp.input_set(0))
    assert(R2 == ncp.input_set(1))
    assert(S1 == ncp.output_set(0))
    assert(S2 == ncp.output_set(1))

def test_ncp_fiedler():
    G = load_example_graph()
    ncp = lgc.NCPData(G)
    ncp.add_neighborhoods()
    ncp.add_fiedler()
    ncp.add_fiedler_mqi()

def test_ncp_crd():
    G = load_example_graph()
    df = lgc.NCPData(G).crd(ratio=1)

def test_ncp_grid():
    import networkx as nx
    K10 = nx.grid_graph(dim=[10,10])
    G = lgc.GraphLocal().from_networkx(K10)
    ncp = lgc.NCPData(G).approxPageRank()
    df = ncp.as_data_frame()
    assert(min(df["output_sizeeff"]) > 0)

def test_ncp_apr():
    G = load_example_graph()
    df = lgc.NCPData(G).approxPageRank(ratio=1)
    df = lgc.NCPData(G).approxPageRank(ratio=2, methodname_prefix="")

def test_ncp_l1reg():
    G = load_example_graph()
    df = lgc.NCPData(G).l1reg(ratio=1)
    print(df)

def test_ncp_localmin():
    G = load_example_graph()
    ncp = lgc.NCPData(G)
    func = lgc.partialfunc(lgc.spectral_clustering,alpha=0.01,rho=1.0e-4,method="acl")

    ncp.default_method = func
    ncp.add_localmin_samples(ratio=1)
    print(ncp.as_data_frame())


    G = lgc.GraphLocal()
    G.list_to_gl([0,1],[1,0],[1,1])
    ncp = lgc.NCPData(G)
    func = lgc.partialfunc(lgc.spectral_clustering,alpha=0.01,rho=1.0e-4,method="acl")

    ncp.default_method = func
    ncp.add_localmin_samples(ratio=1)

def test_ncp_sets():
    G = load_example_graph()
    ncp = lgc.NCPData(G).approxPageRank()
    for i in range(len(ncp.results)):
        R = ncp.input_set(i)
        S = ncp.output_set(i)


def test_apr_deep():
    G = load_example_graph()
    df = lgc.NCPData(G).approxPageRank(ratio=1, gamma=0.1, rholist=[1e-2, 1e-3], deep=True)

def test_apr_only_node_samples():
    G = load_example_graph()
    df = lgc.NCPData(G).approxPageRank(ratio=1, gamma=0.1, rholist=[1e-2, 1e-3], random_neighborhoods=False, localmins=False)

def test_apr_refine():
    G = load_example_graph()
    df = lgc.NCPData(G).approxPageRank(ratio=1, gamma=0.1, rholist=[1e-2, 1e-3],
        random_neighborhoods=False, localmins=False,
        spectral_args={'refine': lgc.partialfunc(lgc.flow_clustering, method="mqi")})

# @pytest.mark.long_tests
# def test_ncp_crd_big():
#     G = lgc.GraphLocal()
#     G.read_graph("notebooks/datasets/minnesota.edgelist","edgelist", remove_whitespace=True)
#     ncp_instance = lgc.NCPData(G)
#     df = ncp_instance.crd(ratio=0.5,w=10,U=10,h=1000,nthreads=4)
#     ncp_plots = lgc.ncpplots.NCPPlots(df)

# @pytest.mark.long_tests
# def test_ncp_l1reg_big():
#     G = lgc.GraphLocal()
#     G.read_graph("notebooks/datasets/neuro-fmri-01.edges","edgelist", " ", header=True)
#     Glcc = G.largest_component()
#     print(Glcc.adjacency_matrix.data)
#     ncp_instance = lgc.NCPData(G)
#     df = ncp_instance.l1reg(ratio=0.5,nthreads=4)

def read_minnesota():
    g = lgc.GraphLocal('notebooks/datasets/minnesota.edgelist','edgelist',' ')
