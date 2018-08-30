import localgraphclustering
import pytest

def load_example_graph():
    return localgraphclustering.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")

def test_ncp():
    G = load_example_graph()
    ncp = localgraphclustering.NCPData(G)
    df = ncp.as_data_frame()
    assert len(df) == 0
    ncp.mqi(nthreads=1,ratio=1.0)
    df = ncp.as_data_frame()
    assert len(df) == G._num_vertices
    func = lambda G,R: localgraphclustering.flow_clustering(G,R,method="mqi")[0]
    ncp = localgraphclustering.NCPData(G)
    ncp.add_set_samples([[1]],nthreads=1,method=func,methodname="mqi")
    ncp.add_random_neighborhood_samples(ratio=2,nthreads=1,method=func,methodname="mqi")



def test_ncp_mqi():
    G = load_example_graph()
    df = localgraphclustering.NCPData(G).mqi(ratio=1)

def test_ncp_crd():
    G = load_example_graph()
    df = localgraphclustering.NCPData(G).crd(ratio=1)

def test_ncp_apr():
    G = load_example_graph()
    df = localgraphclustering.NCPData(G).approxPageRank(ratio=1)

def test_ncp_l1reg():
    G = load_example_graph()
    df = localgraphclustering.NCPData(G).l1reg(ratio=1)
    print(df)

def test_ncp_localmin():
    G = load_example_graph()
    ncp = localgraphclustering.NCPData(G)
    func = lambda G, R: list(spectral_clustering(G,R,alpha=0.01,rho=1.0e-4,method="acl")[0])
    ncp.default_method = func
    ncp.add_localmin_samples(ratio=1)
    print(ncp.as_data_frame())


@pytest.mark.long_tests
def test_ncp_crd_big():
    G = localgraphclustering.GraphLocal()
    G.read_graph("notebooks/datasets/neuro-fmri-01.edges","edgelist", " ", header=True)
    ncp_instance = localgraphclustering.NCPData(G)
    df = ncp_instance.crd(ratio=0.5,w=10,U=10,h=1000,nthreads=1000)
    ncp_plots = localgraphclustering.ncpplots.NCPPlots(df)
    #plot conductance vs size
    #ncp_plots.cond_by_size()
    #plot conductance vs volume
    #ncp_plots.cond_by_vol()
    #plot isoperimetry vs size
    #ncp_plots.isop_by_size()

@pytest.mark.long_tests
def test_ncp_l1reg_big():
    G = localgraphclustering.GraphLocal()
    G.read_graph("notebooks/datasets/neuro-fmri-01.edges","edgelist", " ", header=True)
    Glcc = G.largest_component()
    print(Glcc.adjacency_matrix.data)
    ncp_instance = localgraphclustering.NCPData(G)
    df = ncp_instance.l1reg(ratio=0.5,nthreads=1000)

def read_minnesota():
    g = localgraphclustering.GraphLocal('notebooks/datasets/minnesota.edgelist','edgelist',' ')
