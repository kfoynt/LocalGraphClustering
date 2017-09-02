include("MQI.jl")
include("readSMAT.jl")
include("readSeed.jl")
A=readSMAT("../../graph/minnesota.smat")
R=readSeed("../../graph/minnesota_R.smat")
(actual_length,ret_set)=MQI(A,R)
@show actual_length
@show ret_set
