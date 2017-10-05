/**
 * Compute the densest subgraph given a graph in edge-list format.
 *
 * The edge list format needs to be symmetric.
 *
 * INPUT:
 *     n        - the number of vertices in the graph
 *     ai,aj    - Compressed sparse row representation
 *     offset   - offset for zero based arrays (matlab) or one based arrays (julia)
 *     ret_set  - preallocated memmory the best cluster with the smallest conductance.
 *
 * OUTPUT:
 *     actual_length - the number of nonzero entries in the best set with the largest average degree
 *     ret_set       - the best cluster with the largest average degree
 *     final_degree  - the largest average degree
 *
 * COMPILE:
 *     make densest_subgraph
 */

#ifdef DENSEST_SUBGRAPH_H

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <numeric>
#include "include/routines.hpp"
#include "include/densest_subgraph_c_interface.h"

using namespace std;

template<typename vtype, typename itype>
double graph<vtype,itype>::densest_subgraph(vtype *ret_set, vtype *actual_length)
{
    //cout << "here" << endl;
    vtype nverts, src, dest;
    itype nedges;
    double g, maxflow;
    nverts = n + 2;
    nedges = m + 2 * n;

    double final_degree = 0;//the degree of the densest subgraph
    double L = 0;//lower bound of Andrew Goldberg's algorithm
    double U = m / 2;//upper bound of Andrew Goldberg's algorithm
    vtype *final_cut = (vtype *)malloc(sizeof(vtype) * nverts);

    size_t iter = 0;

    /*malloc enough space to store data used to calculate max flow*/
    vtype *Q = (vtype *)malloc(sizeof(vtype) * nverts);
    vtype *fin = (vtype *)malloc(sizeof(vtype) * nverts);
    vtype *pro = (vtype *)malloc(sizeof(vtype) * nverts);
    vtype *another_pro = (vtype *)malloc(sizeof(vtype) * nverts);
    vtype *pro3 = (vtype *)malloc(sizeof(vtype) * nverts);
    vtype *dist = (vtype *)malloc(sizeof(vtype) * nverts);
    double *flow = (double *)malloc(sizeof(double) * 2 * nedges);
    double *cap = (double *)malloc(sizeof(double) * 2 * nedges);
    vtype *next = (vtype *)malloc(sizeof(vtype) * 2 * nedges);
    vtype *to = (vtype *)malloc(sizeof(vtype) * 2 * nedges);
    vtype *cut = (vtype *)malloc(sizeof(vtype) * nverts);
    double* all_degs = (double*)malloc(sizeof(double) * n);
    //cout << "ok" << endl;

    for(size_t i = 0; i < (size_t)n; i ++){
        all_degs[i] = get_degree_weighted(i);
    }

    /*Andrew Goldberg's algorithm*/
    while(n * (n - 1) * (U - L) >= 1){
        //cout << "ok" << endl;
        iter ++;
        g = (U + L) / 2;
        src = 0;
        dest = nverts - 1;
        //cout << "flow iteration " << iter << ": range = (" << L << ", " << U << "), solution = ";
        maxflow = max_flow_ds<vtype,itype>(ai, aj, a, all_degs, n, m, src, dest, Q, fin, pro, dist,
                                           next, to,cut, another_pro, pro3, flow, cap, g);
        //cout << maxflow << endl;
        if(accumulate(cut, cut + nverts, 0) == 1){
            U = g;
        }
        else{
            L = g;
            for(size_t i = 0; i < (size_t)nverts; i ++){
                final_cut[i] = cut[i];
            }
        }
    }

    final_cut[0] = 0;
    final_cut[nverts - 1] = 0;

    /*retrieve the densest subgraph from the final_cut*/
    vtype num = 0;
    for(size_t i = 1; i < (size_t)nverts - 1; i ++){
        if(final_cut[i] != 0){
            ret_set[num ++] = i - 1;
            //cout << pro3[i] << endl;
            for(vtype &e = pro3[i]; e >= 0; e = next[e]){
                if(final_cut[to[e]] != 0){
                    final_degree += cap[e];
                    //cout << e << " " << cap[e] << endl; 
                }
            }
        }
    }
    final_degree /= (2 * num);
    *actual_length = num;

    /*free space*/
    free(final_cut);
    free(Q);
    free(fin);
    free(pro);
    free(another_pro);
    free(pro3);
    free(dist);
    free(flow);
    free(cap);
    free(next);
    free(to);
    free(cut);
    free(all_degs);
    //cout << "final_degree " << final_degree << endl;
    return final_degree;
}

#endif

