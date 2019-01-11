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
    cout << "In build_map: entering nR loop" << endl;
    for(vtype i = 0; i < nR; i ++){
        R_map[R[i] - offset] = i;
    }
    cout << "In build_map: entering R_map loop" << endl;
    for(auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first;
        deg = get_degree_unweighted(u);
        cout << "In build_map: entering ai loop" << endl;
        cout << "ai[u]" << ai[u] << endl;
        cout << "ai[u+1]" << ai[u+1] << endl;
        cout << "offset" << offset << endl;
        for(vtype j = ai[u] - offset; j < ai[u+1] - offset; j ++){
            cout << "In build_map: entered ai loop" << endl;
            vtype v = aj[j] - offset;
            cout << "In build_map: entered ai loop, set v" << endl;
            if(R_map.count(v) > 0){
                cout << "In build_map: entered ai loop, set R_map.count(v) > 0" << endl;
                deg --;
                cout << "In build_map: entered ai loop, deg --" << endl;
            }
            cout << "In build_map: entered ai loop, for loop end j: " << j << endl;
        }
        cout << "In build_map: for loop R_map end " << endl;
        degree_map[u] = deg;
        cout << "In build_map: degree_map[u] = deg" << endl;
    }
}

template<typename vtype, typename itype>
void graph<vtype,itype>::build_list(unordered_map<vtype, vtype>& R_map, unordered_map<vtype, vtype>& degree_map,
    vtype src, vtype dest, itype A, itype C)
{
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
        vtype u = R_iter->first;
        vtype u1 = R_iter->second;
        for(vtype j = ai[u] - offset; j < ai[u + 1] - offset; j ++){
            vtype v = aj[j] - offset;
            auto got = R_map.find(v);
            if(R_map.count(v) > 0){
                vtype v1 = got->second;
                double w = A;
                addEdge(u1, v1, w);
            }
        }
    }
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
        vtype u1 = src;
        vtype v = R_iter->first;
        vtype v1 = R_iter->second;
        auto got = degree_map.find(v);
        double w = A * got->second;
        addEdge(u1, v1, w);
        //new_edge<vtype,itype>(u1, v1, w, to, cap, flow, next, fin, &nEdge);
        u1 = v1;
        v1 = dest;
        vtype u = v;
        w = C * (ai[u + 1] - ai[u]);
        addEdge(u1, v1, w);
        //new_edge<vtype,itype>(u1, v1, w, to, cap, flow, next, fin, &nEdge);
    }
}


template<typename vtype, typename itype>
vtype graph<vtype,itype>::MQI(vtype nR, vtype* R, vtype* ret_set)
{
    try{
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
        cout << "deg " << total_degree << " cut " << curcutsize << " vol " << curvol << endl;
        condNew = (double)curcutsize/(double)min(total_degree - curvol, curvol);
        cout << "iter: " << total_iter << " conductance: " << condNew << endl;
        total_iter ++;
        vtype nverts = nR + 2;
        adj = new vector<Edge<vtype,itype>>[nverts];
        for (int i = 0; i < nverts; i ++) {
            adj[i].clear();
        }
        level = new int[nverts];
        vector<bool> mincut (nverts);
        build_list(R_map,degree_map,nR,nR+1,curvol,curcutsize);
        pair<double, vtype> retData = DinicMaxflow(nR, nR+1, nverts, mincut);
        delete [] adj;
        delete [] level;
        vtype nRold = nR;
        vtype nRnew = 0; 
        while(condNew < condOld){
            nRnew = nRold - retData.second + 1;
            cout << retData.second << " " << nedges << endl;
            cout << "Krasaro apeira" << endl;
            vtype* Rnew = (vtype*)malloc(sizeof(vtype) * nRnew);
            cout << "Ontws krasaro apeira?" << endl;
            vtype iter = 0;
            for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
                cout << "For loop beginning" << endl;
                vtype u = R_iter->first;
                cout << "u: " << u << endl;
                vtype u1 = R_iter->second;
                cout << "u1: " << u1 << endl;
                cout << "mincut[u1]: " << mincut[u1] << endl;
                if(!mincut[u1]){
                    cout << "Set Rnew[iter] to u + offset: " << (u + offset) << endl;
                    Rnew[iter] = u + offset;
                    cout << "Rnew[iter] has been set to u + offset: " << Rnew << endl;
                    iter ++;
                    cout << "iter incremented" << endl;
                }
            }
            cout << "Setting condOld to condNew" << endl;
            condOld = condNew;
            cout << "Clear R_map" << endl;
            R_map.clear();
            cout << "Clear degree_map" << endl;
            degree_map.clear();
            cout << "Build map" << endl;
            build_map(R_map, degree_map, Rnew, nRnew);
            cout << "End building map" << endl;
            set_stats = get_stats(R_map, nRnew);
            curvol = set_stats.first;
            curcutsize = set_stats.second;
            nedges = curvol - curcutsize + 2 * nRnew;
            cout << "Check if nRnew is > 0: " << nRnew << endl;
            if(nRnew > 0){
                condNew = (double)curcutsize/(double)min(total_degree - curvol, curvol);
                cout << "curvol: " << curvol << " condNew: " << condNew << endl;
                nverts = nRnew + 2;
                cout << "Create new vector" << endl;
                adj = new vector<Edge<vtype,itype>>[nverts];
                cout << "Clear adj matrix" << endl;
                for (int i = 0; i < nverts; i ++) {
                    adj[i].clear();
                }
                level = new int[nverts];
                vector<bool> mincut (nverts);
                cout << "Build list" << endl;
                build_list(R_map,degree_map,nRnew,nRnew+1,curvol,curcutsize);
                cout << "Run DinicMaxflow" << endl;
                retData = DinicMaxflow(nRnew, nRnew + 1, nverts, mincut);
                cout << "End running DinicMaxflow" << endl;
                delete [] adj;
                delete [] level;
                //cout << "here " << nedges << " " << curvol << " " << curcutsize << endl;
                //retData = max_flow_MQI<vtype, itype>(ai, aj, offset, curvol, curcutsize, nedges, nRnew + 2,
                //        R_map, degree_map, nRnew, nRnew + 1, mincut, Q, fin, pro, another_pro, dist, flow, cap, next, to);
            }
            free(Rnew);
            nRold = nRnew;
            cout << "iter: " << total_iter << " conductance: " << condNew << endl;
            total_iter ++;
        }

        //free(mincut);
        vtype j = 0;
        cout << "Start for loop for res_set" << endl;
        cout << "len(ret_set) " << sizeof(*ret_set) << endl;
        for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++ R_iter){
            cout << "ret_set[j] " << ret_set[j] << endl;
            cout << "R_iter->first + offset " << R_iter->first + offset << endl; 
            cout << "Type of first: " << typeid(R_iter->first).name() << endl;
            cout << "Type of offset: " << typeid(offset).name() << endl;
            cout << "Type of ret_set[j]: " << typeid(ret_set[j]).name() << endl;
            ret_set[j] = R_iter->first + offset;
            cout << "ret_set[j] " << ret_set[j] << endl;
            cout << "j in ret_set: " << j << endl;
            j ++;
        } 
        cout << "End for loop for res_set" << endl;
        cout << "Return nRnew: " << nRnew << endl;
        return nRnew;
    }
    catch(const std::exception & ex) {
        cout << "Exception: " << ex.what() << endl;
    }
}

#endif

