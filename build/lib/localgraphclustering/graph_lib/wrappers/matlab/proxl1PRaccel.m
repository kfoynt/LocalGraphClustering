% INPUT:
%     alpha     - teleportation parameter between 0 and 1
%     rho       - l1-reg. parameter
%     ref_node  - seed node
%     ai,aj,a   - Compressed sparse row representation of A
%     d         - vector of node strengths
%     epsilon   - accuracy for termination criterion
%     ds        - the square root of d
%     dsinv     - 1/ds
%     maxiter   - max number of iterations
% 
% OUTPUT:
%     p              - PageRank vector as a row vector
%     not_converged  - flag indicating that maxiter has been reached
%     grad           - last gradient
    
function [not_converged,grad,p]=proxl1PRaccel(A,ref_node,d,ds,dsinv, ...
                                              alpha,rho,epsilon,maxiter,max_time)

[ajPtr,aiPtr,aPtr]=sparse_arrays_ptr(A);
[n,~]=size(A);
grad=zeros(n,1);
p=zeros(n,1);
switch computer
    case {'PCWIN64','GLNXA64','MACI64'}
        indtype = 'int64Ptr';
    case {'PCWIN','GLNX86'}
        indtype = 'uint32Ptr';
    otherwise
        error('Unsupported system');
end
gradPtr=libpointer('doublePtr',grad);
pPtr=libpointer('doublePtr',p);
ref_nodePtr=libpointer(indtype,ref_node);
loadlibrary('../../lib/graph_lib_test/libgraph','../../lib/include/proxl1PRaccel_c_interface.h')
if strcmp(indtype,'int64Ptr')
    not_converged = calllib('libgraph','proxl1PRaccel64',n,aiPtr, ...
                            ajPtr,aPtr,alpha,rho,ref_nodePtr,size(ref_node,2),d,ds,dsinv, ...
                            epsilon,gradPtr,pPtr,maxiter,0,max_time);
else
    not_converged = calllib('libgraph','proxl1PRaccel32',n,aiPtr, ...
                            ajPtr,aPtr,alpha,rho,ref_nodePtr,size(ref_node,2),d,ds,dsinv, ...
                            epsilon,gradPtr,pPtr,maxiter,0,max_time);
end    
grad=get(gradPtr,'Value');
p=get(pPtr,'Value');
unloadlibrary libgraph;
