import pytest

if not pytest.config.getoption("--test-plots"):
    pytest.skip("--test-plots is missing, skipping test of matplotlib", allow_module_level=True)

import localgraphclustering

def load_example_graph():
    return localgraphclustering.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")

def test_ncpplots():
    G = load_example_graph()
    ncp = localgraphclustering.NCPData(G)
    ncp.mqi()
    plots = localgraphclustering.NCPPlots(ncp)
    plots.cond_by_vol()
    plots.cond_by_size()
    plots.isop_by_size()
    plots.mqi_input_output_cond_plot()
    plots.cond_by_vol_itrv(alpha=0.2)
    plots.cond_by_size_itrv(alpha=0.2)
    plots.isop_by_size_itrv(alpha=0.2)
    
    G = load_example_graph()
    ncp = localgraphclustering.NCPData(G)
    ncp.approxPageRank()
    df = ncp.as_data_frame()
    plots = localgraphclustering.NCPPlots(df)
    plots.cond_by_vol()