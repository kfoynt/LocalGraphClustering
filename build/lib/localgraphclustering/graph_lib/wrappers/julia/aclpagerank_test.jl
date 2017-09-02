include("aclpagerank.jl")
include("readSMAT.jl")
include("readSeed.jl")
A=readSMAT("../../graph/Unknown.smat")
alpha=0.99;
eps=10.0^(-7);
seedids=readSeed("../../graph/Unknown_seed.smat");
maxsteps=10000000;
xlength=100;
(actual_length,xids,values)=aclpagerank(A,alpha,eps,seedids,maxsteps,xlength);
@show actual_length
@show xids
@show values
