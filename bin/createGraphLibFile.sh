#!/bin/bash
cd "localgraphclustering/graph_lib/lib/graph_lib_test"
pwd
set -e
make clean
make -f Makefile

cp libgraph.dylib ../../../../build/lib/localgraphclustering/graph_lib/lib/graph_lib_test/

