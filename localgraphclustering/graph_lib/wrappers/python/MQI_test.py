from MQI import MQI
from list_to_CSR import list_to_CSR
import numpy as np

(m,n,ai,aj,a) = list_to_CSR("../../graph/minnesota.smat")

f = open("../../graph/minnesota_R.smat")
data = f.read()
data = data.split()
nR = int(data[0])
R = []
for i in range(nR):
    R += [int(data[i + 1])]
f.close()

(actual_length,results)=MQI(n,ai,aj,nR,R)
print actual_length,results
