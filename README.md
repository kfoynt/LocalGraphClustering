# Local Graph Clustering

Local Graph Clustering provides

- methods that find local clusters in a given graph without touching the whole graph
- methods that improve a given cluster
- methods for global graph partitioning
- tools to compute [Network Community Profiles](http://www.tandfonline.com/doi/abs/10.1080/15427951.2009.10129177)

The current version is 0.4.1 and it is appropriate for experts and intermediates. Contact information for any questions and feedback is given below.

### Authors

- Kimon Fountoulakis, email: kfount at berkeley dot edu
- Meng Liu, email: liu1740 at purdue dot edu
- David Gleich, email: dgleich at purdue dot edu
- Michael Mahoney, email: mmahoney at stat dot berkeley dot edu

## Demonstration

<img src="images/JHopkins.png" width="440" height="250"> <img src="images/Hopkins_global.png" width="440" height="250">
<img src="images/Hopkins_local_1.png" width="440" height="250"> <img src="images/Hopkins_local_2.png" width="440" height="250">

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

It can also be installed through pip:

```
pip3 install localgraphclustering
```

## Import from Julia

1. In Julia, add the PyCall package: 
   
   `Pkg.add(PyCall)`
2. Update which version of Python that PyCall defaults to:

    `ENV["PYTHON"] = (path to python3 executable) `

    `Pkg.build("PyCall")`

    (You can get the path to the python3 executable by just running "which python3" in the terminal.)
3. Make sure the PyPlot package is added in Julia. 
4. Import *localgraphclustering* by using:

   `using PyPlot`

   `@pyimport localgraphclustering`

You can now use any routine in *localgraphluserting* from Julia.

## Examples

All examples are in the [notebooks](https://github.com/kfoynt/LocalGraphClustering/tree/master/notebooks) folder.

Below is a simple demonstration from [test.py](https://github.com/kfoynt/LocalGraphClustering/blob/master/notebooks/test.py) in [notebooks](https://github.com/kfoynt/LocalGraphClustering/tree/master/notebooks) on how to improve spectral partitioning using flow-based methods from local graph clustering.

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

For advanced examples see the Jupyter notebook [here](https://github.com/kfoynt/LocalGraphClustering/blob/master/notebooks/examples.ipynb).

## Examples with visualization

For examples with visualization of the output see the Jupyter notebooks [examples with visualization](https://github.com/kfoynt/LocalGraphClustering/blob/master/notebooks/examples_with_visualization.ipynb), [more example with visualization](https://github.com/kfoynt/LocalGraphClustering/blob/master/notebooks/more_examples_with_visualization.ipynb).

For comparisons of spectral- and flow-based methods with visualization see the Jupyter notebook [here](https://github.com/kfoynt/LocalGraphClustering/blob/master/notebooks/spectral_vs_flow_with_visualization.ipynb).

For visual demonstration of algorithms that can improve a given seed set of nodes see the Jupyter notebook [here](https://github.com/kfoynt/LocalGraphClustering/blob/master/notebooks/improveType_algorithms_with_visualization.ipynb).

## List of applications and methods

-  [Approximate PageRank](https://dl.acm.org/citation.cfm?id=1170528)
- [L1-regularized PageRank](https://link.springer.com/article/10.1007/s10107-017-1214-8) (solved using accelerated proxima gradient descent)
- [PageRank Nibble](https://dl.acm.org/citation.cfm?id=1170528)
- [Rounding methods for spectral embeddings](https://dl.acm.org/citation.cfm?id=1170528)
- [MQI](https://link.springer.com/chapter/10.1007/978-3-540-25960-2_25)
- [FlowImprove](https://dl.acm.org/citation.cfm?id=1347154)
- [SimpleLocal](https://dl.acm.org/citation.cfm?id=3045595)
- [Capacity Releasing Diffusion](http://proceedings.mlr.press/v70/wang17b.html)
- Multiclass label prediction
- [Network Community Profiles](http://www.tandfonline.com/doi/abs/10.1080/15427951.2009.10129177)
- Global spectral partitioning
- Densest subgraph

## License

LocalGraphClustering,
Copyright (C) 2017, Kimon Fountoulakis, Meng Liu, David Gleich, Michael Mahoney

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
