from aclpagerank import aclpagerank
from list_to_CSR import list_to_CSR
import numpy as np

(m,n,ai,aj,a) = list_to_CSR("../../graph/Unknown.smat")

alpha=0.99
eps=10**(-7)
maxsteps=10000000
xlength=100
f = open("../../graph/Unknown_seed.smat")
data = f.read()
data = data.split()
nseedids = int(data[0])
seedids = []
for i in range(nseedids):
    seedids += [data[i + 1]]
f.close()
(actual_length,xids,values)=aclpagerank(n,ai,aj,alpha,eps,seedids,nseedids,maxsteps,xlength,0)
print actual_length,xids,values
