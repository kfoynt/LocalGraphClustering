# A julia wrapper for sweep cut procedure
# A - the sparse matrix representing the symmetric graph
# ids - the order of vertices given
# results - the best set with the smallest conductance
# actual_length - the number of vertices in the best set
# num - the number of vertices given
# values - A vector scoring each vertex (e.g. pagerank value).
#          This will be sorted and turned into one of the other inputs.
# flag - 0 for sweepcut_with_sorting and 1 for sweepcut_without_sorting

const libgraph = joinpath(dirname(@Base.__FILE__),"..","..","lib","graph_lib_test","libgraph")

function sweep_cut{T}(A::SparseMatrixCSC{T,Int64},ids,values,flag,degrees=C_NULL)
    offset=1;
    n=A.n
    idsize=size(ids)
    nids=idsize[1]
    results=zeros(Int64,nids);
    min_cond=[0.0]
    if flag == 1
        actual_length=ccall((:sweepcut_without_sorting64,libgraph),Int64,(Ptr{Int64},
                            Ptr{Int64},Int64,Int64,Ptr{Int64},Ptr{Int64},Ptr{Cdouble},
                            Int64,Ptr{Cdouble},Ptr{Cdouble}),
                            ids,results,nids,n,A.colptr,A.rowval,A.nzval,offset,min_cond,degrees);
    elseif flag == 0
        actual_length=ccall((:sweepcut_with_sorting64,libgraph),Int64,(Ptr{Cdouble},Ptr{Int64},
                            Ptr{Int64},Int64,Int64,Ptr{Int64},Ptr{Int64},Ptr{Cdouble},
                            Int64,Ptr{Cdouble},Ptr{Cdouble}),
                            values,ids,results,nids,n,A.colptr,A.rowval,A.nzval,offset,min_cond,degrees);
    else
        error("Please specify your function (0 for sweepcut_with_sorting and 1 for sweepcut_without_sorting)");
    end
    results=results[1:actual_length]
    return actual_length, results,min_cond[1]
end
