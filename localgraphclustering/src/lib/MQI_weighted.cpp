/**
 * This is an implementation of the MQI algorithm from Lang and Rao (2004) on weighted graphs. 
 * The goal is to find the best subset of a seed set with the smallest conductance.
 *
 * INPUT:
 *     n        - the number of vertices in the graph
 *     ai,aj    - Compressed sparse row representation
 *     offset   - offset for zero based arrays (matlab) or one based arrays (julia)
 *     R        - the seed set
 *     nR       - number of nodes in the original seed set
 *     ret_set  - preallocated memmory for the best cluster with the smallest conductance.
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

#ifdef MQI_WEIGHTED_H

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <stdint.h>
#include <typeinfo>
#include <numeric>      // std::accumulate
#include "include/routines.hpp"
#include "include/MQI_weighted_c_interface.h"
#include <limits>


using namespace std;


template<typename vtype, typename itype>
void graph<vtype,itype>::build_map_weighted(unordered_map<vtype, vtype>& R_map,
                                   unordered_map<vtype, double>& degree_map, vtype* R, vtype nR, 
                                   double* degrees)
{
    double deg;
    for(vtype i = 0; i < nR; i ++){
        //cout << R[i] - offset << " " << i << endl;
        R_map[R[i] - offset] = i;
    }
    for(auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first;
        deg = degrees[u];
        for(vtype j = ai[u] - offset; j < ai[u+1] - offset; j ++){
            vtype v = aj[j] - offset;
            if(R_map.count(v) > 0){
                deg = deg - a[j];
            }
        }
        degree_map[u] = deg;
    }
}

template<typename vtype, typename itype>
void graph<vtype,itype>::build_list_weighted(unordered_map<vtype, vtype>& R_map, unordered_map<vtype, double>& degree_map,
    vtype src, vtype dest, double A, double C, double* degrees)
{
    // replacing edge weight connecting two nodes on side A with A*deg
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
        vtype u = R_iter->first;
        vtype u1 = R_iter->second;
        for(vtype j = ai[u] - offset; j < ai[u + 1] - offset; j ++){
            vtype v = aj[j] - offset;
            auto got = R_map.find(v);
            if(R_map.count(v) > 0){
                vtype v1 = got->second;
                double w = a[j];
                if(v1 > u1) {
                    addEdge(u1, v1, w);
                }
            }
        }
    }

    // add edges from S to node in side B and from node in side B to T
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
        vtype u1 = src;
        vtype v = R_iter->first;
        vtype v1 = R_iter->second;
        auto got = degree_map.find(v);
        double w = got->second;
        addEdge(u1, v1, w);
        u1 = v1;
        v1 = dest;
        vtype u = v;
        w = C/A*degrees[u];
        addEdge(u1, v1, w);
    }
}


template<typename vtype, typename itype>
vtype graph<vtype,itype>::MQI_weighted(vtype nR, vtype* R, vtype* ret_set)
{
    vtype total_iter = 0;
    unordered_map<vtype, vtype> R_map;
    unordered_map<vtype, vtype> old_R_map;
    unordered_map<vtype, double> degree_map;
    build_map_weighted(R_map, degree_map, R, nR, degrees);
    itype nedges = 0;
    double condOld = 1;
    double condNew;
    double total_degree = std::accumulate(degrees,degrees+n,0);
    pair<double, double> set_stats = get_stats_weighted(R_map, nR);
    double curvol = set_stats.first;
    double curcutsize = set_stats.second;
    pair<itype, itype> set_stats_unweighted = get_stats(R_map, nR);
    nedges = set_stats_unweighted.first - set_stats_unweighted.second + 2 * nR;
    //cout << "deg " << total_degree << " cut " << curcutsize << " vol " << curvol << endl;
    if (curvol == 0 || curvol == total_degree) {
        vtype j = 0;
        for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++ R_iter){
            ret_set[j] = R_iter->first + offset;
            j ++;
        }
        return nR;
    }
    condNew = (double)curcutsize/(double)min(total_degree - curvol, curvol);
    //cout << "iter: " << total_iter << " conductance: " << condNew << endl;
    total_iter ++;
    vtype nverts = nR + 2;
    adj = new vector<Edge<vtype,itype>>[nverts];
    for (int i = 0; i < nverts; i ++) {
        adj[i].clear();
    }
    level = new int[nverts];
    vector<bool> mincut (nverts);
    build_list_weighted(R_map,degree_map,nR,nR+1,curvol,curcutsize,degrees);
    pair<double, vtype> retData = DinicMaxflow(nR, nR+1, nverts, mincut);
    //cout << "max flow value: " << retData.first << endl;
    delete [] adj;
    delete [] level;
    vtype nRold = nR;
    vtype nRnew = 0; 
    //cout << condNew << " " << condOld << endl;
    while(condNew < condOld){
        nRnew = nRold - retData.second + 1;
        //cout << retData.second << " " << nedges << endl;
        vtype* Rnew = (vtype*)malloc(sizeof(vtype) * nRnew);
        vtype iter = 0;
        for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
            vtype u = R_iter->first;
            vtype u1 = R_iter->second;
            if(!mincut[u1]){
                Rnew[iter] = u + offset;
                iter ++;
            }
        }
        condOld = condNew;
        old_R_map = R_map;
        R_map.clear();
        degree_map.clear();
        build_map_weighted(R_map, degree_map, Rnew, nRnew, degrees);
        set_stats = get_stats_weighted(R_map, nRnew);
        curvol = set_stats.first;
        curcutsize = set_stats.second;
        set_stats_unweighted = get_stats(R_map, nRnew);
        nedges = set_stats_unweighted.first - set_stats_unweighted.second + 2 * nR;
        //cout << nRnew << " " << old_R_map.size() << endl;
        if(nRnew > 0){
            condNew = (double)curcutsize/(double)min(total_degree - curvol, curvol);
            //cout << "curvol: " << curvol << " condNew: " << condNew << endl;
            nverts = nRnew + 2;
            adj = new vector<Edge<vtype,itype>>[nverts];
            for (int i = 0; i < nverts; i ++) {
                adj[i].clear();
            }
            level = new int[nverts];
            //vector<bool> mincut (nverts);
            build_list_weighted(R_map,degree_map,nRnew,nRnew+1,curvol,curcutsize,degrees);
            retData = DinicMaxflow(nRnew, nRnew + 1, nverts, mincut);
            //cout << "max flow value: " << retData.first << endl;
            delete [] adj;
            delete [] level;
            //cout << "here " << nedges << " " << curvol << " " << curcutsize << endl;
            //retData = max_flow_MQI<vtype, itype>(ai, aj, offset, curvol, curcutsize, nedges, nRnew + 2,
            //        R_map, degree_map, nRnew, nRnew + 1, mincut, Q, fin, pro, another_pro, dist, flow, cap, next, to);
        }
        else {
            vtype j = 0;
            for(auto R_iter = old_R_map.begin(); R_iter != old_R_map.end(); ++ R_iter){
                ret_set[j] = R_iter->first + offset;
                j ++;
            }
            return old_R_map.size();
        }
        free(Rnew);
        nRold = nRnew;
        //cout << "iter: " << total_iter << " conductance: " << condNew << endl;
        total_iter ++;
    }

    //free(mincut);
    vtype j = 0;
    for(auto R_iter = old_R_map.begin(); R_iter != old_R_map.end(); ++ R_iter){
        ret_set[j] = R_iter->first + offset;
        j ++;
    }
    return old_R_map.size();
}

#endif