# A julia wrapper for ppr_path
# A - sparse matrix representation of graph
# alpha - value of alpha
# eps - value of epsilon
# rho - value of rho
# seedids - the set of indices for seeds
# xlength - the max number of ids in the solution vector
# xids, actual_length - the solution vector
# values - the pagerank value vector for xids (already sorted in decreasing order)
type path_info
  num_eps::Ptr{Int64}
  epsilon::Ptr{Cdouble}
  conds::Ptr{Cdouble}
  cuts::Ptr{Cdouble}
  vols::Ptr{Cdouble}
  setsizes::Ptr{Int64}
  stepnums::Ptr{Int64}
end
type rank_info
  starts::Ptr{Int64}
  ends::Ptr{Int64}
  nodes::Ptr{Int64}
  deg_of_pushed::Ptr{Int64}
  size_of_solvec::Ptr{Int64}
  size_of_r::Ptr{Int64}
  val_of_push::Ptr{Cdouble}
  global_bcond::Ptr{Cdouble}
  nrank_changes::Ptr{Int64}
  nrank_inserts::Ptr{Int64}
  nsteps::Ptr{Int64}
  size_for_best_cond::Ptr{Int64}
end

type ret_path_info
  num_eps
  epsilon
  conds
  cuts
  vols
  setsizes
  stepnums
end
type ret_rank_info
  starts
  ends
  nodes
  deg_of_pushed
  size_of_solvec
  size_of_r
  val_of_push
  global_bcond
  nrank_changes
  nrank_inserts
  nsteps
  size_for_best_cond
end

const libgraph = joinpath(dirname(@Base.__FILE__),"..","..","lib","graph_lib_test","libgraph")

function ppr_path{T}(A::SparseMatrixCSC{T,Int64},alpha::Float64,
                          eps::Float64,rho::Float64,seedids,xlength)
    offset = 1;
    n=A.n
    xids=zeros(Int64,xlength);
    seedsize=size(seedids)
    nseedids=seedsize[1]
    maxstep = floor(Int64,1/((1-alpha)*eps))
    num_eps=zeros(Int64,1)
    epsilon=zeros(Cdouble,maxstep)
    conds=zeros(Cdouble,maxstep)
    cuts=zeros(Cdouble,maxstep)
    vols=zeros(Cdouble,maxstep)
    setsizes=zeros(Int64,maxstep)
    stepnums=zeros(Int64,maxstep)
    starts=zeros(Int64,maxstep)
    ends=zeros(Int64,maxstep)
    nodes=zeros(Int64,maxstep)
    deg_of_pushed=zeros(Int64,maxstep)
    size_of_solvec=zeros(Int64,maxstep)
    size_of_r=zeros(Int64,maxstep)
    val_of_push=zeros(Cdouble,maxstep)
    global_bcond=zeros(Cdouble,maxstep)
    nrank_changes=zeros(Int64,1)
    nrank_inserts=zeros(Int64,1)
    nsteps=zeros(Int64,1)
    size_for_best_cond=zeros(Int64,1)
    eps_stats=path_info(pointer(num_eps),pointer(epsilon),pointer(conds),
                            pointer(cuts),pointer(vols),pointer(setsizes),
                            pointer(stepnums))
    rank_stats=rank_info(pointer(starts),pointer(ends),pointer(nodes),pointer(deg_of_pushed),
                             pointer(size_of_solvec),pointer(size_of_r),pointer(val_of_push),pointer(global_bcond),pointer(nrank_changes),
                             pointer(nrank_inserts),pointer(nsteps),pointer(size_for_best_cond))
    actual_length=ccall((:ppr_path64,libgraph),Int64,
                        (Int64,Ptr{Int64},Ptr{Int64},Int64,Cdouble,Cdouble,Cdouble,
                        Ptr{Int64},Int64,Ptr{Int64},Int64,path_info,rank_info),n,A.colptr,A.rowval,
                        offset,alpha,eps,rho,seedids,nseedids,xids,xlength,eps_stats,rank_stats);
    xids=xids[1:actual_length];
    ret_eps_stats=ret_path_info(num_eps[1],epsilon[1:num_eps[1]],conds[1:num_eps[1]],
                              cuts[1:num_eps[1]],vols[1:num_eps[1]],setsizes[1:num_eps[1]],stepnums[1:num_eps[1]])
    ret_rank_stats=ret_rank_info(starts[1:nsteps[1]],ends[1:nsteps[1]],nodes[1:nsteps[1]],deg_of_pushed[1:nsteps[1]],
                                 size_of_solvec[1:nsteps[1]],size_of_r[1:nsteps[1]],val_of_push[1:nsteps[1]],
                                 global_bcond[1:nsteps[1]],nrank_changes[1],nrank_inserts[1],nsteps[1],size_for_best_cond[1])
    return actual_length,xids,ret_eps_stats,ret_rank_stats
end
