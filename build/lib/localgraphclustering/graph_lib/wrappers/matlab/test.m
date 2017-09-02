A=readSMAT('../../graph/Unknown.smat');
A=A+A';
d=sum(A,1)';
ref_node=100;
ds=sqrt(d);
dsinv=1./ds;
alpha=0.15;
rho=1.0e-5;
epsilon=1.0e-4;
maxiter=10000;
max_time=100;
[not_converged,grad,p]=proxl1PRaccel(A,ref_node,d,ds,dsinv, ...
                                              alpha,rho,epsilon,maxiter,max_time);

ids = find(p);
d = sum(A,1)';
sc_p = p(ids)./d(ids);
[actual_length,results]=sweepcut(A,ids-1,sc_p,0);