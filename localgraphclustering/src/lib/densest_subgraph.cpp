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
void graph<vtype,itype>::build_list_DS(double g, vtype src, vtype dest)
{
    for(size_t i = 0; i < (size_t)n; i ++){
        //cout << i << " " << ai[i] << " " << ai[i+1] << endl;
        for(size_t j = ai[i]; j < (size_t)ai[i+1]; j ++){
            //cout << "aj= " << aj[j] << endl;
            //cout << "a= " << a[j] << endl;
            addEdge(i+1,aj[j]+1,a[j]);
        }
    }
    //cout << "start" << endl;
    for(size_t i = 0; i < (size_t)n; i ++){
        addEdge(src,i+1,m*1.0/2);
    }
    //cout << "start" << endl;
    for(size_t i = 0; i < (size_t)n; i ++){
        addEdge(i+1,dest,m*1.0/2+2*g-get_degree_weighted(i));
    }
}


template<typename vtype, typename itype>
double graph<vtype,itype>::densest_subgraph(vtype *ret_set, vtype *actual_length)
{
    //cout << "here" << endl;
    vtype nverts, src, dest;
    double g, maxflow;
    nverts = n + 2;

    double final_degree = 0;//the degree of the densest subgraph
    double L = 0;//lower bound of Andrew Goldberg's algorithm
    double U = m / 2;//upper bound of Andrew Goldberg's algorithm
    vtype *final_cut = (vtype *)malloc(sizeof(vtype) * nverts);

    size_t iter = 0;

    adj = new vector<Edge<vtype,itype>>[nverts];
    level = new int[nverts];
    vector<bool> cut (nverts);
    /*Andrew Goldberg's algorithm*/
    while(n * (n - 1) * (U - L) >= 1){
        //cout << "ok" << endl;
        iter ++;
        g = (U + L) / 2;
        src = 0;
        dest = nverts - 1;


        build_list_DS(g,src,dest);
        pair<double, vtype> retData = DinicMaxflow(src, dest, nverts, cut);
        maxflow = retData.first;
        for (int i = 0; i < nverts; i ++) {
            adj[i].clear();
        }




        if(accumulate(cut.begin(), cut.end(), 0) == 1){
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
            for(vtype j = ai[i-1]; j < ai[i]; j ++){
                if(final_cut[aj[j]+1] != 0){
                    final_degree += a[aj[j]];
                    //cout << e << " " << cap[e] << endl; 
                }
            }
        }
    }
    final_degree /= num;
    *actual_length = num;

    free(final_cut);
    return final_degree;
}

#endif

