from proxl1PRaccel import proxl1PRaccel
import numpy as np
import math
from list_to_CSR import list_to_CSR
'''
ai = np.array([0,2,5,6,8],dtype=np.int64)
aj = np.array([0,3,0,1,3,3,1,3],dtype=np.int64)
a = np.array([1,1,1,1,1,1,1,1],dtype=np.float64)
'''
(m,n,ai,aj,a) = list_to_CSR("../../graph/Unknown_sys.smat")
ref_node = [100,101]
d = np.zeros(n,dtype=np.float64)
ds = np.zeros(n,dtype=np.float64)
dsinv = np.zeros(n,dtype=np.float64)
for i in range(n):
    d[i] = ai[i+1] - ai[i]
    ds[i] = math.sqrt(d[i])
    dsinv[i]  = 1/ds[i]
(not_converged,grad,p) = proxl1PRaccel(ai,aj,a,ref_node,d,ds,dsinv)
print p
print len(p)
