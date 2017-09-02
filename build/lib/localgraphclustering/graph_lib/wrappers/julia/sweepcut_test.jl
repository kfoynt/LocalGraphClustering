include("sweepcut.jl")
include("readSMAT.jl")
include("readSeed.jl")
A=readSMAT("../../graph/minnesota.smat");
ids=readSeed("../../graph/minnesota_ids.smat")
(actual_length,results,min_cond)=sweep_cut(A,ids,~,1);
@show actual_length
@show results
@show min_cond
