# A julia wrapper for weighted aclpagerank
# A - sparse matrix representation of graph
# alpha - value of alpha
# eps - value of epsilon
# seedids - the set of indices for seeds
# maxsteps - the max number of steps
# xlength - the max number of ids in the solution vector
# xids, actual_length - the solution vector
# values - the pagerank value vector for xids (already sorted in decreasing order)

const libgraph = joinpath(dirname(@Base.__FILE__),"..","..","lib","graph_lib_test","libgraph")

function aclpagerank_weighted{T}(A::SparseMatrixCSC{T,Int64},alpha::Float64,
                          eps::Float64,seedids,maxsteps,xlength)
    n=A.n;
    offset=1;
    xids=zeros(Int64,n);
    values=zeros(Cdouble,n);
    seedsize=size(seedids)
    nseedids=seedsize[1]
    actual_length=ccall((:aclpagerank_weighted64,libgraph),Int64,
        (Int64,Ptr{Int64},Ptr{Int64},Ptr{Cdouble},Int64,Cdouble,Cdouble,
        Ptr{Int64},Int64,Int64,Ptr{Int64},Int64,Ptr{Cdouble}),n,A.colptr,
        A.rowval,A.nzval,offset,alpha,eps,seedids,nseedids,maxsteps,xids,xlength,values);
    values=values[1:actual_length];
    xids=xids[1:actual_length];
    return actual_length,xids,values
end
