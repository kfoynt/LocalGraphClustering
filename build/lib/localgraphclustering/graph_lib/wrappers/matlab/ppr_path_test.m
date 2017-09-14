A=readSMAT('../../graph/usps_3nn.smat');
seedids=readSeed('../../graph/usps_3nn_seed.smat');
alpha=0.99;
eps=0.0001;
rho=0.1;
[xlength,~]=size(A);
[actual_length,xids,eps_stats,rank_stats]=ppr_path(A,seedids, ...
                              alpha,eps,rho,xlength);
size(eps_stats.epsilon)
rank_stats.nsteps
rank_stats.nrank_changes