% A matlab wrapper for sweep cut procedure
% A - the sparse matrix representing the symmetric graph
% ids - the order of vertices given
% results - the best set with the smallest conductance
% actual_length - the number of vertices in the best set
% num - the number of vertices given
% values - A vector scoring each vertex (e.g. pagerank value). 
%          This will be sorted and turned into one of the other inputs.
% fun_id - 0 for sweepcut_with_sorting and 1 for sweepcut_without_sorting
% degrees - user defined degrees, set it to be [] if not provided
% min_cond - minimum conductance
function [actual_length,results,min_cond]=sweepcut(A,ids,values,fun_id,degrees)
[ajPtr,aiPtr,aPtr]=sparse_arrays_ptr(A);
[n,~]=size(A);
[num,~]=size(ids);
switch computer
    case {'PCWIN64','GLNXA64','MACI64'}
        indtype = 'int64Ptr';
    case {'PCWIN','GLNX86'}
        indtype = 'uint32Ptr';
    otherwise
        error('Unsupported system');
end
valuesPtr = libpointer('doublePtr',values);
results = zeros(num,1);
resultsPtr = libpointer(indtype,results);
idsPtr = libpointer(indtype,ids);
offset=0;
minCondPtr = libpointer('doublePtr',0.0);
if isempty(degrees)
    degreesPtr = libpointer('doublePtr');
else
    degreesPtr = libpointer('doublePtr',degrees);
end
loadlibrary('../../lib/graph_lib_test/libgraph','../../lib/include/sweepcut_c_interface.h')
if strcmp(indtype,'int64Ptr')
    if fun_id == 1
        actual_length = calllib('libgraph','sweepcut_without_sorting64',idsPtr,...
                                resultsPtr,num,n,aiPtr,ajPtr,aPtr,offset,minCondPtr,degreesPtr);
    elseif fun_id == 0
        actual_length = calllib('libgraph','sweepcut_with_sorting64',valuesPtr,...
                                idsPtr,resultsPtr,num,n,aiPtr,ajPtr,aPtr,offset,minCondPtr,degreesPtr);
    else
        error('Please specify your function (0 for sweepcut_with_sorting and 1 for sweepcut_without_sorting)');
    end
else
    if fun_id == 1
        actual_length = calllib('libgraph','sweepcut_without_sorting32',idsPtr,...
                                resultsPtr,num,n,aiPtr,ajPtr,aPtr,offset,minCondPtr,degreesPtr);
    elseif fun_id == 0
        actual_length = calllib('libgraph','sweepcut_with_sorting32',valuesPtr,...
                                idsPtr,resultsPtr,num,n,aiPtr,ajPtr,aPtr,offset,minCondPtr,degreesPtr);
    else
        error('Please specify your function (0 for sweepcut_with_sorting and 1 for sweepcut_without_sorting)');
    end
end    
results = get(resultsPtr,'Value');
results = results(1:actual_length);
min_cond = get(minCondPtr,'Value');
unloadlibrary libgraph;