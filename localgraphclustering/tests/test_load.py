import localgraphclustering

def test_load():
    G = localgraphclustering.GraphLocal("localgraphclustering/tests/data/dolphins.edges",separator=" ")
    assert G.is_disconnected() == False
