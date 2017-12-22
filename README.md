# Local Graph Clustering

Local Graph Clustering provides

- methods that find local clusters in a given graph without touching the whole graph
- methods that improve a given cluster
- methods for global graph partitioning
- tools to compute [Network Community Profiles](http://www.tandfonline.com/doi/abs/10.1080/15427951.2009.10129177)

## Demonstration

<img src="images/JHopkins.png" width="100" height="100">

![alt text](images/Hopkins_global.png)
![alt text](images/Hopkins_local_1.png)
![alt text](images/Hopkins_local_2.png)

## Installation

```
Clone the repo
```
```
Enter the folder using the termimal
```
```
Type in the terminal `python setup.py install`
```
Note that this package runs only with Python 3.

## Examples

All examples are in the [notebooks](https://github.com/kfoynt/LocalGraphClustering/tree/test_branch/notebooks) folder.

Below is a simple demonstration from [test.py](https://github.com/kfoynt/LocalGraphClustering/blob/test_branch/notebooks/test.py) in [notebooks](https://github.com/kfoynt/LocalGraphClustering/tree/test_branch/notebooks) on how to improve spectral partitioning using flow-based methods from local graph clustering.

```python
from localgraphclustering import *

import time
import numpy as np

# Read graph. This also supports gml and graphml format.
g = graph_class_local.GraphLocal('./datasets/senate.edgelist','edgelist',' ')

# Create an object for global spectral partitioning
sp = spectral_partitioning.Spectral_partitioning()

# Call the global spectral partitioning algorithm.
output_sp = sp.produce([g])

# Only one input graph is given, i.e., [g].
# Extract the array from position 0 and store it.
eig2 = output_sp[0]

# Create an object for the sweep cut rounding procedure.
sc = sweepCut_general.SweepCut_general()

# Round the eigenvector
output_sc = sc.produce([g],p=eig2)

# Extract the partition for g and store it.
eig2_rounded = output_sc[0][0]

# Create an object for subgraph node partitioning.
SL_fast = SimpleLocal_fast.SimpleLocal_fast()

# Conductance before improvement
print("Conductance before improvement:",g.compute_conductance(eig2_rounded))

# Start calling SimpleLocal
start = time.time()
output_SL_fast = SL_fast.produce([g],[eig2_rounded])
end = time.time()
print("running time:",str(end-start)+"s")

# Conductance after improvement
print("Conductance after improvement:",g.compute_conductance(output_SL_fast[0][0]))

output_SL = output_SL_fast[0][0]
```

## Advanced examples

For advanced examples see the Jupyter notebook [here](https://github.com/kfoynt/LocalGraphClustering/blob/test_branch/notebooks/examples.ipynb).

## Examples with visualization

For examples with visualization of the output see the Jupyter notebooks [examples with visualization](https://github.com/kfoynt/LocalGraphClustering/blob/test_branch/notebooks/examples_with_visualization.ipynb), [more example with visualization](https://github.com/kfoynt/LocalGraphClustering/blob/test_branch/notebooks/more_examples_with_visualization.ipynb).

For a comparison of spectral- and flow-based methods with visualization see the Jupyter notebook [here](https://github.com/kfoynt/LocalGraphClustering/blob/test_branch/notebooks/spectral_vs_flow_with_visualization.ipynb).

For a visual demonstration of algorithms that can improve a given seed set of nodes see the Jupyter notebook [here](https://github.com/kfoynt/LocalGraphClustering/blob/test_branch/notebooks/improveType_algorithms_with_visualization.ipynb).

## List of applications and methods

- Approximate PageRank
- L1-regularized PageRank (solved using accelerated proxima gradient descent)
- PageRank Nibble
- Rounding methods for spectral embeddings
- MQI
- FlowImprove
- SimpleLocal
- Capacity Releasing Diffusion
- Multiclass label prediction
- Network Community Profiles
- Global spectral partitioning
- Densest subgraph
