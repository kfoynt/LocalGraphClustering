
"""
These functions implement a more simplified interface to our local
tools than the more general functional interfaces. They should be
used when simplicity is key.
"""


from localgraphclustering import capacity_releasing_diffusion_fast
from localgraphclustering import MQI_fast
from localgraphclustering import l1_regularized_PageRank_fast
from localgraphclustering import sweepCut_fast
from localgraphclustering import approximate_PageRank_Clustering

def simple_crd(G,R,U=3,h=10,w=2,iterations=20):
    crd_fast = capacity_releasing_diffusion_fast.Capacity_Releasing_Diffusion_fast()
    return list(crd_fast.produce([G],R,U,h,w,iterations)[0])

def simple_mqi(G,R):
    MQI_fast_obj = MQI_fast.MQI_fast()
    output_MQI_fast = MQI_fast_obj.produce([G],[R])
    return output_MQI_fast[0][0].tolist()

## TODO, write this!
def simple_flow_improve(G,R):
    pass

def simple_l1_diffusion(G,R,alpha,rho,epsilon=1.0e-1,iterations=1000):
    l1reg_fast = l1_regularized_PageRank_fast.L1_regularized_PageRank_fast()
    sc_fast = sweepCut_fast.SweepCut_fast()
    output_l1reg_fast = l1reg_fast.produce([G],R,alpha=alpha,rho=rho,epsilon=epsilon,iterations=iterations)
    return sc_fast.produce([G],p=output_l1reg_fast[0])[0][0].tolist()

def simple_approx_pagerank(G,R,alpha,rho,iterations=10000):
    pr_clustering = approximate_PageRank_Clustering.Approximate_PageRank_Clustering()
    output_pr_clustering = pr_clustering.produce([G],R,alpha=alpha,rho=rho,iterations=iterations)
    return output_pr_clustering[0]

