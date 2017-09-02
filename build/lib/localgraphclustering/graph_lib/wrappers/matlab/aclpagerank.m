% A matlab wrapper for aclpagerank
% A - sparse matrix representation of graph
% alpha - value of alpha
% eps - value of epsilon
% seedids,nseedids - the set of indices for seeds
% maxsteps - the max number of steps
% xlength - the max number of ids in the solution vector
% xids, actual_length - the solution vector
% values - the pagerank value vector for xids (already sorted in decreasing order)
function [actual_length,xids,values]=aclpagerank(A,seedids, ...
                                                    alpha,eps,maxsteps,xlength)
[ajPtr,aiPtr,~]=sparse_arrays_ptr(A);
[n,~]=size(A);
[nseedids,~]=size(seedids);
x=zeros(xlength,1);
switch computer
    case {'PCWIN64','GLNXA64','MACI64'}
        indtype = 'int64Ptr';
    case {'PCWIN','GLNX86'}
        indtype = 'uint32Ptr';
    otherwise
        error('Unsupported system');
end
values=zeros(xlength,1);
xPtr = libpointer(indtype,x);
valuePtr = libpointer('doublePtr',values);
seedPtr = libpointer(indtype,seedids);
loadlibrary('../../lib/graph_lib_test/libgraph','../../lib/include/aclpagerank_c_interface.h')
if strcmp(indtype,'int64Ptr')
    actual_length = calllib('libgraph','aclpagerank64',n,aiPtr, ...
                            ajPtr,0,alpha,eps,seedPtr,nseedids,maxsteps, ...
                            xPtr,xlength,valuePtr);
else
    actual_length = calllib('libgraph','aclpagerank32',n,aiPtr, ...
                            ajPtr,0,alpha,eps,seedPtr,nseedids,maxsteps, ...
                            xPtr,xlength,valuePtr);
end    
xids=get(xPtr,'Value');
values=get(valuePtr,'Value');
xids=xids(1:actual_length);
values=values(1:actual_length);
unloadlibrary libgraph;
