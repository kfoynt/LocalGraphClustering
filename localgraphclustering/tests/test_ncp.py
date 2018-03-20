import localgraphclustering

def load_example_graph():
    return localgraphclustering.graph_class_local.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")

def test_ncp():
    G = load_example_graph()
    ncp = localgraphclustering.ncp.NCPData(G)
    df = ncp.as_data_frame()
    assert len(df) == 0
    ncp.default_method = lambda G,R: localgraphclustering.simple_interface.simple_mqi(G,R)
    ncp.add_random_neighborhood_samples(nthreads=1,ratio=1.0)
    df = ncp.as_data_frame()
    assert len(df) == G._num_vertices
    
    ncp = localgraphclustering.ncp.NCPData(G)
    ncp.default_method = lambda G,R: localgraphclustering.simple_interface.simple_mqi(G,R)
    ncp.add_set_samples([[1]],nthreads=1)
    
    
    
def test_ncp_mqi():
    G = load_example_graph()
    df = localgraphclustering.ncp.Ncp().produce(G,"mqi",ratio=1)
    
def test_ncp_crd():
    G = load_example_graph()
    df = localgraphclustering.ncp.Ncp().produce(G,"crd",ratio=1)    

def test_ncp_apr():
    G = load_example_graph()
    df = localgraphclustering.ncp.Ncp().produce(G,"approxPageRank",ratio=1)    
    
def test_ncp_l1reg():
    G = load_example_graph()
    df = localgraphclustering.ncp.Ncp().produce(G,"l1reg",ratio=1)    
    print(df)
    
