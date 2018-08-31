import pytest

if not pytest.config.getoption("--test-plots"):
    pytest.skip("--test-plots is missing, skipping test of matplotlib", allow_module_level=True)

import localgraphclustering
import matplotlib.pyplot as plt

def load_example_graph():
    return localgraphclustering.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")

def test_ncpplots():
    G = localgraphclustering.graph_class_local("localgraphclustering/tests/data/dolphins.edges",separator=" ")
    ncp = localgraphclustering.NCPData(G).mqi()
    plots = localgraphclustering.NCPPlots(ncp)
    plots.mqi_input_output_cond_plot()
    plots.cond_by_vol()
    plots.cond_by_size()
    plots.isop_by_size()

    df = localgraphclustering.NCPData(G).approxPageRank().as_data_frame()
    plots = localgraphclustering.NCPPlots(df)
    plots.cond_by_vol()
