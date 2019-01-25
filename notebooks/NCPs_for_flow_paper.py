import localgraphclustering as lgc

import time
import numpy as np

# Import matplotlib 
import matplotlib.pyplot as plt
    
import sys, traceback
import os
sys.path.insert(0, os.path.join("..", "LocalGraphClustering", "notebooks"))
import helper
import pickle
import csv

# print("Running senate")

# # Read graph. This also supports gml and graphml format.
# g = lgc.GraphLocal('./datasets/senate.graphml','graphml')
# g.discard_weights()

# ncp_instance = lgc.NCPData(g)
# ncp_instance.approxPageRank(ratio=0.8,timeout=5000000,nthreads=24)

# ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
# #plot conductance vs size
# fig, ax, min_tuples = ncp_plots.cond_by_size()
# plt.savefig('figures/cond_card_senate.png', bbox_inches='tight')
# plt.show()
# #plot conductance vs volume
# fig, ax, min_tuples = ncp_plots.cond_by_vol()
# plt.savefig('figures/cond_vol_senate.png', bbox_inches='tight')
# plt.show()
# #plot isoperimetry vs size
# fig, ax, min_tuples = ncp_plots.isop_by_size()
# plt.savefig('figures/expand_card_senate.png', bbox_inches='tight')
# plt.show()

# pickle.dump(ncp_instance, open('results/' + "acl" + "-ncp-" + "senate" + '.pickle', 'wb'))

# print("Running JohnsHopkins")

# # Read John Hopkins graph.
# g = lgc.GraphLocal('./datasets/JohnsHopkins.graphml','graphml')
# g.discard_weights()

# ncp_instance = lgc.NCPData(g)
# ncp_instance.approxPageRank(ratio=0.8,timeout=5000000,nthreads=24)

# ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
# #plot conductance vs size
# fig, ax, min_tuples = ncp_plots.cond_by_size()
# plt.savefig('figures/cond_card_Johns.png', bbox_inches='tight')
# plt.show()
# #plot conductance vs volume
# fig, ax, min_tuples = ncp_plots.cond_by_vol()
# plt.savefig('figures/cond_vol_Johns.png', bbox_inches='tight')
# plt.show()
# #plot isoperimetry vs size
# fig, ax, min_tuples = ncp_plots.isop_by_size()
# plt.savefig('figures/expand_card_Johns.png', bbox_inches='tight')
# plt.show()

# pickle.dump(ncp_instance, open('results/' + "acl" + "-ncp-" + "JohnsHopkins" + '.pickle', 'wb'))

# print("Running usroads")

# # Read John Hopkins graph.
# g = lgc.GraphLocal('./datasets/usroads-cc.graphml','graphml')
# g.discard_weights()

# ncp_instance = lgc.NCPData(g)
# ncp_instance.approxPageRank(ratio=0.5,timeout=5000000,nthreads=24)

# ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
# #plot conductance vs size
# fig, ax, min_tuples = ncp_plots.cond_by_size()
# plt.savefig('figures/cond_card_usroad.png', bbox_inches='tight')
# plt.show()
# #plot conductance vs volume
# fig, ax, min_tuples = ncp_plots.cond_by_vol()
# plt.savefig('figures/cond_vol_usroad.png', bbox_inches='tight')
# plt.show()
# #plot isoperimetry vs size
# fig, ax, min_tuples = ncp_plots.isop_by_size()
# plt.savefig('figures/expand_card_usroad.png', bbox_inches='tight')
# plt.show()

# pickle.dump(ncp_instance, open('results/' + "acl" + "-ncp-" + "usroads" + '.pickle', 'wb'))

# print("Running Colgate")

# # Read John Hopkins graph.
# g = lgc.GraphLocal('./datasets/Colgate88_reduced.graphml','graphml')
# g.discard_weights()

# ncp_instance = lgc.NCPData(g)
# ncp_instance.approxPageRank(ratio=0.8,timeout=5000000,nthreads=24)

# ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
# #plot conductance vs size
# fig, ax, min_tuples = ncp_plots.cond_by_size()
# plt.savefig('figures/cond_card_colgate.png', bbox_inches='tight')
# plt.show()
# #plot conductance vs volume
# fig, ax, min_tuples = ncp_plots.cond_by_vol()
# plt.savefig('figures/cond_vol_colgate.png', bbox_inches='tight')
# plt.show()
# #plot isoperimetry vs size
# fig, ax, min_tuples = ncp_plots.isop_by_size()
# plt.savefig('figures/expand_card_colgate.png', bbox_inches='tight')
# plt.show()

# pickle.dump(ncp_instance, open('results/' + "acl" + "-ncp-" + "colgate" + '.pickle', 'wb'))

print("Running ppimips")

# Read graph. This also supports gml and graphml format.
# The MIPS Mammalian Protein-Protein Database is a database for protein-protein interactions of mammalian species. 
# We used the data set proposed in consisting of a subset of 220 protein complexes of 1562 proteins. 
# Details can be found here: https://clusteval.sdu.dk/1/datasets/685
g = lgc.GraphLocal('./datasets/ppi_mips.graphml','graphml',' ')
g.discard_weights()

ncp_instance = lgc.NCPData(g)
ncp_instance.approxPageRank(ratio=1,timeout=5000000,nthreads=24)

ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
#plot conductance vs size
fig, ax, min_tuples = ncp_plots.cond_by_size()
plt.savefig('figures/cond_card_ppimips.png', bbox_inches='tight')
plt.show()
#plot conductance vs volume
fig, ax, min_tuples = ncp_plots.cond_by_vol()
plt.savefig('figures/cond_vol_ppimips.png', bbox_inches='tight')
plt.show()
#plot isoperimetry vs size
fig, ax, min_tuples = ncp_plots.isop_by_size()
plt.savefig('figures/expand_card_ppimips.png', bbox_inches='tight')
plt.show()

pickle.dump(ncp_instance, open('results/' + "acl" + "-ncp-" + "ppimips" + '.pickle', 'wb'))

print("Running sfld")

# Read graph. This also supports gml and graphml format.
# The data set contains pairwise similarities of blasted 
# sequences of 232 proteins belonging to the amidohydrolase superfamily. 
g = lgc.GraphLocal('./datasets/sfld_brown_et_al_amidohydrolases_protein_similarities_for_beh.graphml','graphml',' ')
g.discard_weights()

ncp_instance = lgc.NCPData(g)
ncp_instance.approxPageRank(ratio=1,timeout=5000000,nthreads=24)

ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
#plot conductance vs size
fig, ax, min_tuples = ncp_plots.cond_by_size()
plt.savefig('figures/cond_card_sfld.png', bbox_inches='tight')
plt.show()
#plot conductance vs volume
fig, ax, min_tuples = ncp_plots.cond_by_vol()
plt.savefig('figures/cond_vol_sfld.png', bbox_inches='tight')
plt.show()
#plot isoperimetry vs size
fig, ax, min_tuples = ncp_plots.isop_by_size()
plt.savefig('figures/expand_card_sfld.png', bbox_inches='tight')
plt.show()

pickle.dump(ncp_instance, open('results/' + "acl" + "-ncp-" + "sfld" + '.pickle', 'wb'))

# mygraphs = {'email-Enron':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/email-Enron.edgelist',
#             'pokec':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/soc-pokec-relationships.edgelist',
#             'orkut':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/com-orkut.ungraph.edgelist',
#             'livejournal':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/soc-LiveJournal1.edgelist'
#            }

# start = time.time()
# for (gname,gfile) in mygraphs.items():
#     print(gname, gfile)
#     sep = ' '
#     if isinstance(gfile, tuple):
#         sep = gfile[1]
#         gfile = gfile[0]
        
#     print("Running " + gname)
    
#     g = lgc.GraphLocal(os.path.join("..", "data", gfile),'edgelist', "	")
#     g.discard_weights()

#     ncp_instance = lgc.NCPData(g)
#     ncp_instance.approxPageRank(ratio=0.3,timeout=5000000,nthreads=24)

#     ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
#     #plot conductance vs size
#     fig, ax, min_tuples = ncp_plots.cond_by_size()
#     plt.savefig('figures/cond_card_' + gname + '.png', bbox_inches='tight')
#     plt.show()
#     #plot conductance vs volume
#     fig, ax, min_tuples = ncp_plots.cond_by_vol()
#     plt.savefig('figures/cond_vol_' + gname + '.png', bbox_inches='tight')
#     plt.show()
#     #plot isoperimetry vs size
#     fig, ax, min_tuples = ncp_plots.isop_by_size()
#     plt.savefig('figures/expand_card_' + gname + '.png', bbox_inches='tight')
#     plt.show()
    
#     pickle.dump(ncp_instance, open('results/' + "acl" + "-ncpNOTIMPROVE-" + gname + '.pickle', 'wb'))