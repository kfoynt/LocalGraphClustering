import localgraphclustering as lgc
import numpy as np

import matplotlib.pyplot as plt


import sys, traceback
import os
sys.path.insert(0, os.path.join("..", "LocalGraphClustering", "notebooks"))
import helper
import pickle

import time

np.random.seed(seed=123)

helper.lgc_graphlist

def run_improve(g, gname, method, methodname, delta, nthreads=24, timeout=1000):
    ratio = 1.0
    if g._num_vertices > 1000000:
        ratio = 0.05
    elif g._num_vertices > 100000:
        ratio = 0.1    
    elif g._num_vertices > 10000:
        ratio = 0.4        
    elif g._num_vertices > 7500:
        ratio = 0.6
    elif g._num_vertices > 5000:
        ratio = 0.8
    print("ratio: ", ratio)
    start = time.time()
    ncp = lgc.NCPData(g,store_output_clusters=True)
    ncp.approxPageRank(ratio=ratio,nthreads=nthreads,localmins=False,neighborhoods=False,random_neighborhoods=False)
    end = time.time()
    print("Elapsed time for acl-ncp for dataset ", gname , " is ", end - start, " the method is ", methodname, " delta is ", delta)
    sets = [st["output_cluster"] for st in ncp.results]
    
    print("Make an NCP object for Improve Algo")
    start2 = time.time()
    ncp2 = lgc.NCPData(g)
    print("Going into improve mode")
    output = ncp2.refine(sets, method=method, methodname=methodname, nthreads=nthreads, timeout=timeout, **{"delta": delta})
    end2 = time.time()
    print("Elapsed time for improve-ncp for dataset ", gname , " is ", end2 - start2, " the method is ", methodname, " delta is ", delta)
    fig = lgc.NCPPlots(ncp2).mqi_input_output_cond_plot()[0]
    fig.axes[0].set_title(gname + " " + methodname+"-NCP")
    fig.savefig("figures/" + method + "-ncp-"+gname+".pdf", bbox_inches="tight", figsize=(100,100))
    plt.show()
    pickle.dump(ncp, open('results/' + method + "-ncp-" + gname + '.pickle', 'wb'))
    ncp.write('results/' + method + "-ncp-csv-" + gname, writepython=False)
    pickle.dump(ncp2, open('results/' + method + "-ncp2-" + gname + '.pickle', 'wb'))
    ncp2.write('results/' + method + "-ncp2-csv-" + gname, writepython=False)
    
    delta = 0.3
    method="sl" 
    methodname="SimpleLocal"
    
    print("Make an NCP object for Improve Algo")
    start2 = time.time()
    ncp2 = lgc.NCPData(g)
    print("Going into improve mode")
    output = ncp2.refine(sets, method=method, methodname=methodname, nthreads=nthreads, timeout=timeout, **{"delta": delta})
    end2 = time.time()
    print("Elapsed time for improve-ncp for dataset ", gname , " is ", end2 - start2, " the method is ", methodname, " delta is ", delta)
    fig = lgc.NCPPlots(ncp2).mqi_input_output_cond_plot()[0]
    fig.axes[0].set_title(gname + " " + methodname+"-NCP")
    fig.savefig("figures/" + method + "delta" + str(delta) + "-ncp-"+gname+".pdf", bbox_inches="tight", figsize=(100,100))
    plt.show()
    pickle.dump(ncp, open('results/' + method + "delta" + str(delta) + "-ncp-" + gname + '.pickle', 'wb'))
    ncp.write('results/' + method + "delta" + str(delta) + "-ncp-csv-" + gname, writepython=False)
    pickle.dump(ncp2, open('results/' + method + "delta" + str(delta) + "-ncp2-" + gname + '.pickle', 'wb'))
    ncp2.write('results/' + method + "delta" + str(delta) + "-ncp2-csv-" + gname, writepython=False)
    
    delta = 0.6
    method="sl" 
    methodname="SimpleLocal"
    
    print("Make an NCP object for Improve Algo")
    start2 = time.time()
    ncp2 = lgc.NCPData(g)
    print("Going into improve mode")
    output = ncp2.refine(sets, method=method, methodname=methodname, nthreads=nthreads, timeout=timeout, **{"delta": delta})
    end2 = time.time()
    print("Elapsed time for improve-ncp for dataset ", gname , " is ", end2 - start2, " the method is ", methodname, " delta is ", delta)
    fig = lgc.NCPPlots(ncp2).mqi_input_output_cond_plot()[0]
    fig.axes[0].set_title(gname + " " + methodname+"-NCP")
    fig.savefig("figures/" + method + "delta" + str(delta) + "-ncp-"+gname+".pdf", bbox_inches="tight", figsize=(100,100))
    plt.show()
    pickle.dump(ncp, open('results/' + method + "delta" + str(delta) + "-ncp-" + gname + '.pickle', 'wb'))
    ncp.write('results/' + method + "delta" + str(delta) + "-ncp-csv-" + gname, writepython=False)
    pickle.dump(ncp2, open('results/' + method + "delta" + str(delta) + "-ncp2-" + gname + '.pickle', 'wb'))
    ncp2.write('results/' + method + "delta" + str(delta) + "-ncp2-csv-" + gname, writepython=False)
    
    delta = 0.9
    method="sl" 
    methodname="SimpleLocal"
    
    print("Make an NCP object for Improve Algo")
    start2 = time.time()
    ncp2 = lgc.NCPData(g)
    print("Going into improve mode")
    output = ncp2.refine(sets, method=method, methodname=methodname, nthreads=nthreads, timeout=timeout, **{"delta": delta})
    end2 = time.time()
    print("Elapsed time for improve-ncp for dataset ", gname , " is ", end2 - start2, " the method is ", methodname, " delta is ", delta)
    fig = lgc.NCPPlots(ncp2).mqi_input_output_cond_plot()[0]
    fig.axes[0].set_title(gname + " " + methodname+"-NCP")
    fig.savefig("figures/" + method + "delta" + str(delta) + "-ncp-"+gname+".pdf", bbox_inches="tight", figsize=(100,100))
    plt.show()
    pickle.dump(ncp, open('results/' + method + "delta" + str(delta) + "-ncp-" + gname + '.pickle', 'wb'))
    ncp.write('results/' + method + "delta" + str(delta) + "-ncp-csv-" + gname, writepython=False)
    pickle.dump(ncp2, open('results/' + method + "delta" + str(delta) + "-ncp2-" + gname + '.pickle', 'wb'))
    ncp2.write('results/' + method + "delta" + str(delta) + "-ncp2-csv-" + gname, writepython=False)
    
mygraphs = {#'email-Enron':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/email-Enron.edgelist',
            #'pokec':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/soc-pokec-relationships.edgelist',
            'orkut':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/com-orkut.ungraph.edgelist',
            'livejournal':'/u4/kfountoulakis/flowReviewPaper/LocalGraphClustering/notebooks/datasets/soc-LiveJournal1.edgelist'
           }

for (gname,gfile) in mygraphs.items():
    print(gname, gfile)
    sep = ' '
    if isinstance(gfile, tuple):
        sep = gfile[1]
        gfile = gfile[0]
    g = lgc.GraphLocal(os.path.join("..", "data", gfile),'edgelist', "	")
    g.discard_weights()
    run_improve(g, gname=gname, method="mqi", methodname="MQI", delta=100, timeout=100000000)