from sweepcut import sweepcut
from list_to_CSR import list_to_CSR
import numpy as np

(m,n,ai,aj,a) = list_to_CSR("../../graph/minnesota.smat")

f = open("../../graph/minnesota_ids.smat")
data = f.read()
data = data.split()
nids = int(data[0])
ids = []
for i in range(nids):
    ids += [int(data[i + 1])]
f.close()

values=[]
flag = 1
degrees = [0.0]*n
for i in range(n):
    for j in range(ai[i],ai[i+1]):
        degrees[i] += a[j]
(actual_length,bestclus,min_cond)=sweepcut(n,ai,aj,a,ids,nids,values,flag,degrees=degrees)
print actual_length,bestclus,min_cond
(actual_length,bestclus,min_cond)=sweepcut(n,ai,aj,a,ids,nids,values,flag)
print actual_length,bestclus,min_cond
