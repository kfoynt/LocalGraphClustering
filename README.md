Local Graph Clustering
======================

Local Graph Clustering provides methods to find local clusters in a given graph
without touching the whole graph.  

Installation
============

Clone the repo, enter the folder using the termimal and type in the terminal `python setup.py install`. 
Alternatively, it can be intalled using pip by typing in the terminal `pip install localgraphclustering`.
Note that this package runs only with python3.

Import from Julia
=================
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

Examples
========

In the "notebooks" folder see the Jupyter notebooks "examples", "examples_with_visualization", 
"more_local_examples_with_visualization", "spectral_vs_flow_with_visualization" and 
"improve_algorithms_with_visualization".