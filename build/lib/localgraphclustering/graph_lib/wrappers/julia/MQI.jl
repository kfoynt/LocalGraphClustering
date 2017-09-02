# A julia wrapper for MQI
# A - sparse matrix representation of graph
# R - seed set
# actual_length - number of nodes in the optimal subset
# ret_set - optimal subset with the smallest conductance

const libgraph = joinpath(dirname(@Base.__FILE__),"..","..","lib","graph_lib_test","libgraph")

function MQI{T}(A::SparseMatrixCSC{T,Int64},R)
    Rsize=size(R)
    nR=Rsize[1]
    ret_set=zeros(Int64,nR);
    n=A.n
    offset = 1;
    actual_length=ccall((:MQI64,libgraph),Int64,
                        (Int64,Int64,Ptr{Int64},Ptr{Int64},Int64,Ptr{Int64},Ptr{Int64}),n,nR,A.colptr,
                        A.rowval,offset,R,ret_set);
    ret_set = ret_set[1:actual_length]
    return actual_length,ret_set
end
