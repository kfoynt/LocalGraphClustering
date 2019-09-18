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

# pickle.dump(ncp_instance, open('results/ncp-senate.pickle', 'wb'))

# # Run us-roads
# g = lgc.GraphLocal('./datasets/usroads-cc.graphml','graphml')
# g.discard_weights()

# ncp_instance = lgc.NCPData(g)
# ncp_instance.approxPageRank(ratio=0.8,timeout=5000000,nthreads=24)

# ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
# #plot conductance vs size
# fig, ax, min_tuples = ncp_plots.cond_by_size()
# plt.savefig('figures/cond_card_usroads.png', bbox_inches='tight')
# plt.show()
# #plot conductance vs volume
# fig, ax, min_tuples = ncp_plots.cond_by_vol()
# plt.savefig('figures/cond_vol_usroads.png', bbox_inches='tight')
# plt.show()
# #plot isoperimetry vs size
# fig, ax, min_tuples = ncp_plots.isop_by_size()
# plt.savefig('figures/expand_card_usroads.png', bbox_inches='tight')
# plt.show()

# pickle.dump(ncp_instance, open('results/ncp-usroads.pickle', 'wb'))

mygraphs = {#'email-Enron':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/email-Enron.edgelist',
            #'pokec':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/soc-pokec-relationships.edgelist',
            #'orkut':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/com-orkut.ungraph.edgelist',
            'livejournal':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/soc-LiveJournal1.edgelist'
           }

for (gname,gfile) in mygraphs.items():
    print(gname, gfile)
    sep = ' '
    if isinstance(gfile, tuple):
        sep = gfile[1]
        gfile = gfile[0]
        
    print("Running " + gname)
    
    g = lgc.GraphLocal(os.path.join("..", "data", gfile),'edgelist', "	")
    g.discard_weights()

    start = time.time()
    
    ncp_instance = lgc.NCPData(g)
    ncp_instance.approxPageRank(ratio=0.1,timeout=5000000,nthreads=24)

    ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "acl")
    #plot conductance vs size
    fig, ax, min_tuples = ncp_plots.cond_by_size()
    plt.savefig('figures/cond_card_' + gname + '.png', bbox_inches='tight')
    plt.show()
    #plot conductance vs volume
    fig, ax, min_tuples = ncp_plots.cond_by_vol()
    plt.savefig('figures/cond_vol_' + gname + '.png', bbox_inches='tight')
    plt.show()
    #plot isoperimetry vs size
    fig, ax, min_tuples = ncp_plots.isop_by_size()
    plt.savefig('figures/expand_card_' + gname + '.png', bbox_inches='tight')
    plt.show()
    
    end = time.time()
    print(" Elapsed time: ", end - start)
    
    pickle.dump(ncp_instance, open('results/ncp' + gname + '.pickle', 'wb'))
    ncp.write('results/ncp' + gname, writepython=False)
    
    
# Run orkut seperately
print("Running orkut")

name = '/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/com-orkut.ungraph.edgelist'
g = lgc.GraphLocal(os.path.join(data_path,name),'edgelist', "	")

comm_name = '/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/com-orkut.top5000.cmty.txt'
ref_nodes_unfiltered = []
with open(comm_name, "r") as f:
    for line in f:
        new_line = []
        for i in line.split():
            if i.isdigit():
                new_line.append(int(i))
        ref_nodes_unfiltered.append(new_line)
     
    
n = g._num_vertices

number_feature = 0

ref_nodes = []
info_ref_nodes = []

for ff in ref_nodes_unfiltered:

    vol_ff = sum(g.d[ff])

    if vol_ff < 100:
        continue

    cond_ff = g.compute_conductance(ff,cpp=True)

    if cond_ff > 0.47:
        continue

    print("Reached")
    eig_ff, lambda_ff = lgc.fiedler_local(g, ff)
    lambda_ff = np.real(lambda_ff)
    gap_ff = lambda_ff/cond_ff

    print("Number of feature", number_feature, " gap ",gap_ff, " volume: ", vol_ff, " size:", len(ff), "conductance: ", cond_ff)

    if gap_ff >= 0.5 and vol_ff >= 100:
        ref_nodes.append(ff)
        np.save('results/ref_nodes_orkut', ref_nodes) 
        
    number_feature += 1

start = time.time()

ncp_instance = lgc.NCPData(g)
ncp_instance.approxPageRank(ratio=0.1,timeout=5000000,nthreads=24)
ncp_instance.add_set_samples_without_method(ref_nodes)

ncp_plots = lgc.NCPPlots(ncp_instance,method_name = "")

end = time.time()
print(" Elapsed time: ", end - start)
    
#plot conductance vs size
fig, ax, min_tuples = ncp_plots.cond_by_size()
counter = 0
for cluster in ref_nodes:
    ax.scatter([len(cluster)], [g.compute_conductance(cluster,cpp=True)], c="green", s=250, marker='D',zorder=100000)
    counter += 1
plt.savefig('figures/cond_card_orkut.png', bbox_inches='tight')
plt.show()
#plot conductance vs volume
fig, ax, min_tuples = ncp_plots.cond_by_vol()
plt.savefig('figures/cond_vol_orkut.png', bbox_inches='tight')
plt.show()
#plot isoperimetry vs size
fig, ax, min_tuples = ncp_plots.isop_by_size()
plt.savefig('figures/expand_card_orkut.png', bbox_inches='tight')
plt.show()

pickle.dump(ncp_instance, open('results/ncporkut.pickle', 'wb'))

ncp.write('results/ncporkut' + gname, writepython=False)
