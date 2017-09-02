A=readSMAT('../../graph/minnesota_weighted.smat');
A=A';
seedids=readSeed('../../graph/minnesota_weighted_seed.smat');
alpha=0.99;
eps=10^(-7);
maxsteps=10000000;
xlength=100;
[actual_length,xids,values]=aclpagerank_weighted(A,seedids,alpha,eps, ...
                                           maxsteps,xlength);
actual_length
xids
values