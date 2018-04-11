import os
from localgraphclustering import graph_class_local
def lgc_data(name):
    #lgc_path = os.path.join("..", "LocalGraphClustering")
    lgc_path = os.path.dirname(os.path.realpath(__file__))
    if name=="senate":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "senate.edgelist"), 'edgelist',  ' ')
    elif name=="Erdos02":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "Erdos02-cc.edgelist"), 'edgelist',  ' ')
    elif name=="JohnsHopkins":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "JohnsHopkins.edgelist"), 'edgelist',  '\t')
    elif name=="Colgate88":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "Colgate88_reduced.graphml"), 'graphml')
    elif name=="usroads":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "usroads-cc.edgelist"), 'edgelist',  ' ')
    elif name=="ppi_mips":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "ppi_mips.graphml"), 'graphml')
    elif name=="ASTRAL":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", 
                    "ASTRAL-small-sized-mammoth-sims-geq-2.graphml"), 'graphml')
    elif name=="sfld":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", 
                    "sfld_brown_et_al_amidohydrolases_protein_similarities_for_beh.graphml"), 'graphml')
    elif name=="find_V":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "find_V.graphml"), 'graphml')
    elif name=="ppi-homo":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "ppi-homo.edgelist"), 'edgelist', ' ')
    elif name=="neuro-fmri-01":
        return graph_class_local(os.path.join(
                lgc_path, "datasets", "neuro-fmri-01.edges"), 'edgelist', ' ')
    else:
        raise Exception("Unknown graph name")

lgc_graphlist = ["senate", "Erdos02", "JohnsHopkins", "Colgate88", 
    "usroads", "ppi_mips", "ASTRAL", "sfld", "find_V", "ppi-homo", "neuro-fmri-01"]
