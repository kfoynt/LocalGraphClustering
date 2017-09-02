/**
 * This is an implementation of the MQI algorithm from Lang and Rao (2004). 
 * The goal is to find the best subset of a seed set with the smallest conductance.
 *
 * INPUT:
 *     n        - the number of vertices in the graph
 *     ai,aj    - Compressed sparse row representation
 *     offset   - offset for zero based arrays (matlab) or one based arrays (julia)
 *     R        - the seed set
 *     nR       - number of nodes in the original seed set
 *     ret_set  - preallocated memmory the best cluster with the smallest conductance.
 *
 * OUTPUT:
 *     actual_length - the number of nonzero entries in the best set with the lowest conductance
 *     ret_set       - the best cluster with the smallest conductance.
 *
 * COMPILE:
 *     make MQI
 *
 * EXAMPLE:
 *     Use functions from readData.hpp to read a graph and seed from files.
 *     int64_t* ret_set = (int64_t*)malloc(sizeof(int64_t) * nR);
 *     int64_t offset = 0;
 *     int64_t actual_length = MQI64(m, nR, ai, aj, offset, R, ret_set);
 */

#ifdef MQI_H

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <stdint.h>
#include <typeinfo>
#include "include/routines.hpp"
#include "include/MQI_c_interface.h"

using namespace std;


    template<typename vtype, typename itype>
void graph<vtype,itype>::build_map(unordered_map<vtype, vtype>& R_map,
                                   unordered_map<vtype, vtype>& degree_map, vtype* R, vtype nR)
{
    vtype deg;
    for(vtype i = 0; i < nR; i ++){
        R_map[R[i] - offset] = i;
    }
    for(auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first;
        deg = get_degree_unweighted(u);
        for(vtype j = ai[u] - offset; j < ai[u+1] - offset; j ++){
            vtype v = aj[j] - offset;
            if(R_map.count(v) > 0){
                deg --;
            }
        }
        degree_map[u] = deg;
    }
}


    template<typename vtype, typename itype>
vtype graph<vtype,itype>::MQI(vtype nR, vtype* R, vtype* ret_set)
{
    vtype total_iter = 0;
    unordered_map<vtype, vtype> R_map;
    unordered_map<vtype, vtype> degree_map;
    build_map(R_map, degree_map, R, nR);
    itype nedges = 0;
    double condOld = 1;
    double condNew;
    itype total_degree = ai[n] - offset;
    pair<itype, itype> set_stats = get_stats(R_map, nR);
    itype curvol = set_stats.first;
    itype curcutsize = set_stats.second;
    nedges = curvol - curcutsize + 2 * nR;
    //cout << "deg " << total_degree << " cut " << curcutsize << " vol " << curvol << endl;
    condNew = (double)curcutsize/(double)min(total_degree - curvol, curvol);
    cout << "iter: " << total_iter << " conductance: " << condNew << endl;
    total_iter ++;
    vtype* mincut = (vtype*)malloc(sizeof(vtype) * (nR + 2));
    pair<double, vtype> retData = max_flow<vtype, itype>(ai, aj, offset, curvol, curcutsize, nedges,
            nR + 2, R_map, degree_map, nR, nR + 1, mincut);
    vtype nRold = nR;
    vtype nRnew; 
    while(condNew < condOld){
        nRnew = nRold - retData.second + 1;
        //cout << nRnew << " " << nedges << endl;
        vtype* Rnew = (vtype*)malloc(sizeof(vtype) * nRnew);
        vtype iter = 0;
        for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
            vtype u = R_iter->first;
            vtype u1 = R_iter->second;
            if(mincut[u1] == 0){
                Rnew[iter] = u + offset;
                iter ++;
            }
        }
        condOld = condNew;
        R_map.clear();
        degree_map.clear();
        build_map(R_map, degree_map, Rnew, nRnew);
        set_stats = get_stats(R_map, nRnew);
        curvol = set_stats.first;
        curcutsize = set_stats.second;
        nedges = curvol - curcutsize + 2 * nRnew;
        if(nRnew > 0){
            condNew = (double)curcutsize/(double)min(total_degree - curvol, curvol);
            //cout << "here" << nedges << " " << curvol << " " << curcutsize << endl;
            retData = max_flow<vtype, itype>(ai, aj, offset, curvol, curcutsize, nedges, nRnew + 2,
                    R_map, degree_map, nRnew, nRnew + 1, mincut);
        }
        free(Rnew);
        nRold = nRnew;
        cout << "iter: " << total_iter << " conductance: " << condNew << endl;
        total_iter ++;
    }

    free(mincut);
    vtype j = 0;
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++ R_iter){
        ret_set[j] = R_iter->first + offset;
        j ++;
    }
    return nRnew;
}

#endif

