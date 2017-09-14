# INPUT:
#     alpha     - teleportation parameter between 0 and 1
#     rho       - l1-reg. parameter
#     ref_node  - seed node
#     ai,aj,a   - Compressed sparse row representation of A
#     d         - vector of node strengths
#     epsilon   - accuracy for termination criterion
#     ds        - the square root of d
#     dsinv     - 1/ds
#     maxiter   - max number of iterations
# 
# OUTPUT:
#     p              - PageRank vector as a row vector
#     not_converged  - flag indicating that maxiter has been reached
#     grad           - last gradient


const libgraph = joinpath(dirname(@Base.__FILE__),"..","..","lib","graph_lib_test","libgraph")

function proxl1PRaccel{T}(A::SparseMatrixCSC{T,Int64},ref_node,d,ds,dsinv;alpha=0.15,
                        rho=1.0e-5,epsilon=1.0e-4,maxiter=10000,max_time=100)
    n=A.n;
    offset=1;
    grad=zeros(Cdouble,n);
    p=zeros(Cdouble,n);
    if typeof(ref_node) == Int64
        ref_node = [ref_node]
    end
    not_converged=ccall((:proxl1PRaccel64,libgraph),Int64,
                        (Int64,Ptr{Int64},Ptr{Int64},Ptr{Cdouble},Cdouble,Cdouble,
                         Ptr{Int64},Int64,Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Cdouble,Ptr{Cdouble},Ptr{Cdouble},
                         Int64,Int64,Int64),n,A.colptr,A.rowval,A.nzval,alpha,rho,ref_node,length(ref_node),
                        d,ds,dsinv,epsilon,grad,p,maxiter, offset,max_time);
    return not_converged,grad,p
end
