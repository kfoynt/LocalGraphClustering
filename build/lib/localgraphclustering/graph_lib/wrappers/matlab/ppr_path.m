% A matlab wrapper for aclpagerank
% A - sparse matrix representation of graph
% alpha - value of alpha
% eps - value of epsilon
% seedids,nseedids - the set of indices for seeds
% maxsteps - the max number of steps
% xlength - the max number of ids in the solution vector
% xids, actual_length - the solution vector
% values - the pagerank vector for xids (already sort decreasing order)
function [actual_length,xids,eps_stats,rank_stats]=ppr_path(A,seedids, ...
                                        alpha,eps,rho,xlength)
[ajPtr,aiPtr,~]=sparse_arrays_ptr(A);
[n,~]=size(A);
[nseedids,~]=size(seedids);
x=zeros(xlength,1);
switch computer
    case {'PCWIN64','GLNXA64','MACI64'}
        indtype = 'int64Ptr';
        max_step=int64(1/((1-alpha)*eps));
    case {'PCWIN','GLNX86'}
        indtype = 'uint32Ptr';
        max_step=int32(1/((1-alpha)*eps));
    otherwise
        error('Unsupported system');
end


num_eps = 0;
epsilon = zeros(max_step,1);
conds = zeros(max_step,1);
cuts = zeros(max_step,1);
vols = zeros(max_step,1);
setsizes = zeros(max_step,1,'int64');
stepnums = zeros(max_step,1,'int64');
path_info.num_eps = libpointer('int64Ptr',num_eps);
path_info.epsilon = libpointer('doublePtr',epsilon);
path_info.conds = libpointer('doublePtr',conds);
path_info.cuts = libpointer('doublePtr',cuts);
path_info.vols = libpointer('doublePtr',vols);
path_info.setsizes = libpointer('int64Ptr',setsizes);
path_info.stepnums = libpointer('int64Ptr',stepnums);

nrank_changes = 0;
nrank_inserts = 0;
nsteps = 0;
size_for_best_cond = 0;
starts = zeros(max_step,1);
ends = zeros(max_step,1);
nodes = zeros(max_step,1);
deg_of_pushed = zeros(max_step,1);
size_of_solvec = zeros(max_step,1);
size_of_r = zeros(max_step,1);
val_of_push = zeros(max_step,1);
global_bcond = zeros(max_step,1);
rank_info.nrank_changes = libpointer('int64Ptr',nrank_changes);
rank_info.nrank_inserts = libpointer('int64Ptr',nrank_inserts);
rank_info.nsteps = libpointer('int64Ptr',nsteps);
rank_info.size_for_best_cond = libpointer('int64Ptr',size_for_best_cond);
rank_info.starts = libpointer('int64Ptr',starts);
rank_info.ends = libpointer('int64Ptr',ends);
rank_info.nodes = libpointer('int64Ptr',nodes);
rank_info.deg_of_pushed = libpointer('int64Ptr',deg_of_pushed);
rank_info.size_of_solvec = libpointer('int64Ptr',size_of_solvec);
rank_info.size_of_r = libpointer('int64Ptr',size_of_r);
rank_info.val_of_push = libpointer('doublePtr',val_of_push);
rank_info.global_bcond = libpointer('doublePtr',global_bcond);


xPtr = libpointer(indtype,x);
seedPtr = libpointer(indtype,seedids);
loadlibrary('../../lib/graph_lib_test/libgraph','../../lib/include/ppr_path_c_interface.h')
if strcmp(indtype,'int64Ptr')
    actual_length = calllib('libgraph','ppr_path64',n,aiPtr, ...
                            ajPtr,0,alpha,eps,rho,seedPtr,nseedids, ...
                            xPtr,xlength,path_info,rank_info);
else
    actual_length = calllib('libgraph','ppr_path32',n,aiPtr, ...
                            ajPtr,0,alpha,eps,rho,seedPtr,nseedids, ...
                            xPtr,xlength,path_info,rank_info);
end    
xids=get(xPtr,'Value');
xids=xids(1:actual_length);


num_eps=get(path_info.num_eps,'Value');
epsilon=get(path_info.epsilon,'Value');
conds=get(path_info.conds,'Value');
cuts=get(path_info.cuts,'Value');
vols=get(path_info.vols,'Value');
setsizes=get(path_info.setsizes,'Value');
stepnums=get(path_info.stepnums,'Value');
eps_stats.num_eps=num_eps;
eps_stats.epsilon=epsilon(1:num_eps);
eps_stats.conds=conds(1:num_eps);
eps_stats.cuts=cuts(1:num_eps);
eps_stats.vols=vols(1:num_eps);
eps_stats.setsizes=setsizes(1:num_eps);
eps_stats.stepnums=stepnums(1:num_eps);

nrank_changes = get(rank_info.nrank_changes,'Value');
nrank_inserts = get(rank_info.nrank_inserts,'Value');
nsteps = get(rank_info.nsteps,'Value');
size_for_best_cond = get(rank_info.size_for_best_cond,'Value');
starts = get(rank_info.starts,'Value');
ends = get(rank_info.ends,'Value');
nodes = get(rank_info.nodes,'Value');
deg_of_pushed = get(rank_info.deg_of_pushed,'Value');
size_of_solvec = get(rank_info.size_of_solvec,'Value');
size_of_r = get(rank_info.size_of_r,'Value');
val_of_push = get(rank_info.val_of_push,'Value');
global_bcond = get(rank_info.global_bcond,'Value');
rank_stats.nrank_changes = nrank_changes;
rank_stats.nrank_inserts = nrank_inserts;
rank_stats.nsteps = nsteps;
rank_stats.size_for_best_cond = size_for_best_cond;
rank_stats.starts = starts(1:nsteps);
rank_stats.ends = ends(1:nsteps);
rank_stats.nodes = nodes(1:nsteps);
rank_stats.deg_of_pushed = deg_of_pushed(1:nsteps);
rank_stats.size_of_solvec = size_of_solvec(1:nsteps);
rank_stats.size_of_r = size_of_r(1:nsteps);
rank_stats.val_of_push = val_of_push(1:nsteps);
rank_stats.global_bcond = global_bcond(1:nsteps);


unloadlibrary libgraph;