## graph_lib
A graph library with C interface and julia, python and matlab wrappers.

## Current routines
### aclpagerank
* ai,aj,offset - Compressed sparse row representation, with offset for zero based (matlab) on one based arrays (julia)
* alpha - value of alpha
* eps - value of epsilon
* seedids,nseedids - the set of indices for seeds
* maxsteps - the max number of steps
* xlength - the max number of ids in the solution vector
* xids, actual_length - the solution vector
* values - the pagerank value vector for xids (already sorted in decreasing order)
### sweepcut
* ai,aj,offset - Compressed sparse row representation, with offset for zero based (matlab) on one based arrays (julia)
* ids - the order of vertices given
* results - the best set with the smallest conductance
* actual_length - the number of vertices in the best set
* num - the number of vertices given
* values - A vector scoring each vertex (e.g. pagerank value). This will be sorted and turned into one of the other inputs.

This interface contains 6 functions :

sweepcut\_with\_sorting64

sweepcut\_with\_sorting32 

sweepcut\_with\_sorting32_64

sweepcut\_without\_sorting64

sweepcut\_without\_sorting32 

sweepcut\_without\_sorting32_64

"64" means all data is in 64bit, "32" means all data is in 32bit and "32_64" means "ai" is in 64bit while the others are in 32bit.

The first three functions will sort "ids" based on the decreasing order of "values", while the other three don't have this step.
### MQI
 This is an implementation of the MQI algorithm from Lang and Rao (2004). The goal is to find the best subset of a seed set with the smallest conductance.
 
 * n - number of nodes in the graph
 * nR - number of nodes in the original seed set
 * ai,aj,offset - the CSR representation of graph with offset for zero based (matlab) and one based arrays (julia)
 * R - the seed set
 * ret_set - the best subset
 * actual_length - the number of nodes in ret_set

### ppr_path
 * n, ai, aj, offset - Compressed sparse row representation, 
                            with offset for zero based (matlab) or 
                            one based arrays (julia)
 * alpha - value of alpha
 * eps - value of epsilon
 * rho - value of rho
 * seedids - indices of seed set
 * nseedids - number of indices in seed set
 * xids, xlength - the solution vector

## Usage
Inside the lib/graph\_lib\_test folder, use the following command to create the dynamic library and all the executable files,
	
	make
	
use the following command to run all the tests,

	python test_all.py
	
use the following command to delete all the generated files,

	make clean
	
