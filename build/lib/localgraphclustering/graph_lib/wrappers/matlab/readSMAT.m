function A = readSMAT(filename)
  
% READSMAT Load a graph into a Matlab sparse matrix
 
% A = readSMAT(filename) where
 
% filename is the name of the SMAT file and
 
% A is the MATLAB sparse matrix
 
 
if ~exist(filename,'file')
 
error('readSMAT:fileNotFound', 'Unable to read file %s', filename);
 
end
 
 
s = load(filename,'-ascii');
 
m = s(1,1);

n = s(1,2);
 
 
ind_i = s(2:length(s),1)+1;
 
ind_j = s(2:length(s),2)+1;

val = s(2:length(s),3);

clear s;
 
A = sparse(ind_i,ind_j,val, m, n);
