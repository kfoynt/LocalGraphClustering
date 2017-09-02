include("aclpagerank_weighted.jl")
include("readSMAT.jl")
include("readSeed.jl")
A=readSMAT("../../graph/minnesota_weighted.smat")
A=A'
alpha=0.99;
eps=10.0^(-7);
seedids=readSeed("../../graph/minnesota_weighted_seed.smat");
maxsteps=10000000;
xlength=100;
(actual_length,xids,values)=aclpagerank_weighted(A,alpha,eps,seedids,maxsteps,xlength);
@show actual_length
@show xids
@show values
