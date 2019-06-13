import os
import localgraphclustering as lgc
def lgc_data(name):
    #lgc_path = os.path.join("..", "LocalGraphClustering")
    lgc_path = os.path.dirname(os.path.realpath(__file__))
    if name=="senate":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "senate.edgelist"), 'edgelist',  ' ')
    elif name=="Erdos02":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "Erdos02-cc.edgelist"), 'edgelist',  ' ')
    elif name=="dolphins":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "dolphins.smat"), 'edgelist',  ' ')
    elif name=="JohnsHopkins":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "JohnsHopkins.edgelist"), 'edgelist',  '\t')
    elif name=="Colgate88":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "Colgate88_reduced.graphml"), 'graphml')
    elif name=="usroads":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "usroads-cc.edgelist"), 'edgelist',  ' ')
    elif name=="ppi_mips":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "ppi_mips.graphml"), 'graphml')
    elif name=="ASTRAL":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets",
                    "ASTRAL-small-sized-mammoth-sims-geq-2.graphml"), 'graphml')
    elif name=="sfld":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets",
                    "sfld_brown_et_al_amidohydrolases_protein_similarities_for_beh.graphml"), 'graphml')
    elif name=="find_V":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "find_V.graphml"), 'graphml')
    elif name=="ppi-homo":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "ppi-homo.edgelist"), 'edgelist', ' ', header=True)
    elif name=="neuro-fmri-01":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "neuro-fmri-01.edges"), 'edgelist', ' ', header=True)
    elif name=="ca-GrQc":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "ca-GrQc-cc.csv"), 'edgelist', ' ', header=True)
    elif name=="disconnected":
        return lgc.GraphLocal(os.path.join(
                lgc_path, "../../notebooks/datasets", "disconnected.smat"), 'edgelist',  ' ')
    else:
        raise Exception("Unknown graph name")

lgc_graphlist = ["senate", "Erdos02", "JohnsHopkins", "Colgate88",
    "usroads", "ppi_mips", "ASTRAL", "sfld", "find_V", "ppi-homo",
    "neuro-fmri-01", "ca-GrQc","dolphins","disconnected"]
