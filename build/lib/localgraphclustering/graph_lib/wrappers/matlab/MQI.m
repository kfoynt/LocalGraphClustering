% A matlab wrapper for MQI
% A - sparse matrix representation of graph
% n - number of nodes in the graph
% R - seed set
% nR - number of nodes in seed set
% actual_length - number of nodes in the optimal subset
% ret_set - optimal subset with the smallest conductance
function [actual_length,ret_set]=MQI(A,R)
[ajPtr,aiPtr,~]=sparse_arrays_ptr(A);
[n,~]=size(A);
[nR,~]=size(R);
switch computer
    case {'PCWIN64','GLNXA64','MACI64'}
        indtype = 'int64Ptr';
    case {'PCWIN','GLNX86'}
        indtype = 'uint32Ptr';
    otherwise
        error('Unsupported system');
end
ret_set=zeros(nR,1);
ret_setPtr = libpointer(indtype,ret_set);
RPtr = libpointer(indtype,R);
loadlibrary('../../lib/graph_lib_test/libgraph','../../lib/include/MQI_c_interface.h')
if strcmp(indtype,'int64Ptr')
    actual_length = calllib('libgraph','MQI64',...
                            n,nR,aiPtr,ajPtr,0,RPtr,ret_setPtr);
else
    actual_length = calllib('libgraph','MQI32',...
                            n,nR,aiPtr,ajPtr,0,RPtr,ret_setPtr);
end    
ret_set=get(ret_setPtr,'Value');
ret_set=ret_set(1:actual_length);
unloadlibrary libgraph;