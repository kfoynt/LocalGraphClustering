include("ppr_path.jl")
include("readSMAT.jl")
include("readSeed.jl")
A=readSMAT("../../graph/usps_3nn.smat")
seedids=readSeed("../../graph/usps_3nn_seed.smat")
alpha = 0.99
eps = 0.0001
rho = 0.1
seedids = [7576]
xlength=A.n
(actual_length,xids,ret_eps_stats,ret_rank_stats)=ppr_path(A,alpha,eps,rho,seedids,xlength)
@show actual_length
@show xids
