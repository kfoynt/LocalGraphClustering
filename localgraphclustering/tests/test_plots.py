import pytest

if not pytest.config.getoption("--test-plots"):
    pytest.skip("--test-plots is missing, skipping test of matplotlib", allow_module_level=True)

import localgraphclustering
import matplotlib.pyplot as plt

def load_example_graph():
    return localgraphclustering.graph_class_local.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")

def test_ncpplots():
    G = localgraphclustering.graph_class_local.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")
    ncp = localgraphclustering.ncp.Ncp().produce(G,method="mqi")
    plots = localgraphclustering.ncpplots.NCPPlots(ncp)
    plots.mqi_input_output_cond_plot()
    plots.cond_by_vol()
    plots.cond_by_size()
    plots.isop_by_size()
    
    df = localgraphclustering.ncp.Ncp().produce(G,method="approxPageRank").as_data_frame()
    plots = localgraphclustering.ncpplots.NCPPlots(df)
    plots.cond_by_vol()

    

