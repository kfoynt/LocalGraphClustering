function [pir,pjc,pa] = sparse_arrays_ptr(A)
% SPARSE_ARRAYS_PTR Return the internal matlab sparse arrays as pointers
%
% [ir,jc,a] = sparse_arrays_ptr(A) returns the raw internal
% representations of the Matlab arrays as pointers that can 
% base passed to other places, such as other libraries
%
% We do some system checking, which does cause some slight
% overhead (probably more than getting the arrays...)
% so it may be worth caching them if you use them repeatedly.
%
% Example: 
% % This example shows getting an array and scaling the values
% % without making a copy by calling a function in one of the
% % test libraries.
% if not(libisloaded('shrlibsample'))
%     loadlibrary(...
%         fullfile(matlabroot,'extern','examples','shrlib',['shrlibsample.' mexext]),...
%         fullfile(matlabroot,'extern','examples','shrlib','shrlibsample.h'))
% end
% C = sprand(4,8,0.5);
% [~,~,pa] = sparse_arrays_ptr(C);
% C
% calllib('shrlibsample','multDoubleArray',pa,5)
% C

assert(issparse(A))
switch computer
    case {'PCWIN64','GLNXA64','MACI64'}
        indtype = 'int64Ptr';
    case {'PCWIN','GLNX86'}
        indtype = 'uint32Ptr';
    otherwise
        error('Unsupported system');
end
n = size(A,2);
nz = nnz(A);
if not(libisloaded('libmx'))
    hfile = fullfile(matlabroot,'extern','include','matrix.h');
    loadlibrary('libmx',hfile);
end    
[pir,~] = calllib('libmx','mxGetIr_730', A);
[pjc,~] = calllib('libmx','mxGetJc_730', A);
if nargout > 2
    [pa,~] = calllib('libmx','mxGetJc_730', A);
end
setdatatype(pir,indtype,nz);
setdatatype(pjc,indtype,n+1);
if nargout > 2
    [pa,~] = calllib('libmx','mxGetPr', A);
    if isa(A,'double')
        setdatatype(pa,'doublePtr',nz)
    elseif isa(A,'logical')
        setdatatype(pa,'logicalPtr',nz)
    else
        error('unsupported datatype');
    end
end