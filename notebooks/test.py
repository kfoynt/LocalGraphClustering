from localgraphclustering import *

import time
import numpy as np

# Read graph. This also supports gml and graphml format.
g = GraphLocal('./datasets/senate.edgelist','edgelist',' ')

# Call the global spectral partitioning algorithm.
eig2 = fiedler(g)

# Round the eigenvector
output_sc = sweep_cut(g,eig2)

# Extract the partition for g and store it.
eig2_rounded = output_sc[0]

# Conductance before improvement
print("Conductance before improvement:",g.compute_conductance(eig2_rounded))

# Start calling SimpleLocal
start = time.time()
output_SL_fast = SimpleLocal(g,eig2_rounded)
end = time.time()
print("running time:",str(end-start)+"s")

# Conductance after improvement
print("Conductance after improvement:",g.compute_conductance(output_SL_fast[0]))

output_SL = output_SL_fast[0]
