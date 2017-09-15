from aclpagerank import aclpagerank
from aclpagerank_weighted import aclpagerank_weighted
from list_to_CSR import list_to_CSR
from MQI import MQI
from ppr_path import ppr_path
from proxl1PRaccel import proxl1PRaccel
from sweepcut import sweepcut

class GRAPH(object):
    def list_to_CSR(self,filename):
        return list_to_CSR(filename)
    def aclpagerank(self,n,ai,aj,alpha,eps,seedids,nseedids,maxsteps,xlength,flag):
        return aclpagerank(n,ai,aj,alpha,eps,seedids,nseedids,maxsteps,xlength,flag)
    def aclpagerank_weighted(self,n,ai,aj,a,alpha,eps,seedids,nseedids,maxsteps,xlength,offset):
        return aclpagerank_weighted(n,ai,aj,a,alpha,eps,seedids,nseedids,maxsteps,xlength,offset)
    def MQI(self,n,ai,aj,nR,R):
        return MQI(n,ai,aj,nR,R)
    def ppr_path(self,n,ai,aj,alpha,eps,rho,seedids,nseedids,xlength):
        return ppr_path(n,ai,aj,alpha,eps,rho,seedids,nseedids,xlength)
    def proxl1PRaccel(self,ai,aj,a,ref_node,d,ds,dsinv,alpha = 0.15,rho = 1.0e-5,epsilon = 1.0e-4,maxiter = 10000,max_time = 100):
        return proxl1PRaccel(ai,aj,a,ref_node,d,ds,dsinv,alpha = 0.15,rho = 1.0e-5,epsilon = 1.0e-4,maxiter = 10000,max_time = 1000)
    def sweepcut(self,n,ai,aj,a,ids,num,values,flag,degrees = None):
        return sweepcut(n,ai,aj,a,ids,num,values,flag,degrees = None)
