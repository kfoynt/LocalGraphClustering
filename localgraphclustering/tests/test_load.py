import localgraphclustering as lgc
import networkx as nx
import pytest

def test_load():
    G = lgc.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")
    assert G.is_disconnected() == False

def test_from_networkx():
    import math
    import numpy as np
    N = 1000
    rad = 3/math.sqrt(N)
    np.random.seed(0)
    pos_x = np.random.rand(N)
    pos_y = np.random.rand(N)
    pos = {i: (pos_x[i], pos_y[i]) for i in range(N)}
    G = nx.generators.random_geometric_graph(N,rad,pos=pos)
    g = lgc.GraphLocal().from_networkx(G)
    assert (g.adjacency_matrix != g.adjacency_matrix.T).sum() == 0
    # On our test system, this returns true, could need to be adjusted
    # due to the random positions.
    assert g.is_disconnected() == False



@pytest.mark.long_tests
def test_load_datasets():
    import os
    import sys
    mypath = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(mypath,"..","..","notebooks"))
    import helper

    for gname in helper.lgc_graphlist:
        g = helper.lgc_data(gname)

    sys.path.pop(0)
