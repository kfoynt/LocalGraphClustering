#ifdef SIMPLELOCAL_H

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <stdint.h>
#include <typeinfo>
#include <functional>
#include "include/routines.hpp"
#include "include/SimpleLocal_c_interface.h"
#include <tuple>

using namespace std;

template<typename vtype, typename itype>
void graph<vtype,itype>::init_VL(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,
                                 unordered_map<vtype,vtype>& R_map, vtype s, vtype t)
{
    vtype num = 0;
    VL[s] = num;
    VL_rev[num] = s;
    num ++;
    VL[t] = num;
    VL_rev[num] = t;
    num ++;
    for (auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first;
        if (VL.count(u) == 0) {
            VL[u] = num;
            VL_rev[num] = u;
            num ++;
        }
        for (itype j = ai[u-2]; j < ai[u-2+1]; j ++) {
            vtype v = aj[j]+2;
            if (VL.count(v) == 0) {
                VL[v] = num;
                VL_rev[num] = v;
                num ++;
            }
        }
    }
}

template<typename vtype, typename itype>
void graph<vtype,itype>::init_EL(vector<tuple<vtype,vtype,double>>& EL, unordered_map<vtype,vtype>& R_map,
                                 vtype s, vtype t, double alpha, double beta)
{
    cout << "alpha " << alpha << " beta " << beta << endl;
    //unordered_map<vtype,vtype> R_map;
    unordered_map<vtype,vtype> B_map;
    vtype ARR  = 0;
    vtype ABR  = 0;
    for (auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first+2;
        EL.push_back(make_tuple(s,u,get_degree_unweighted(u-2)*alpha));
        EL.push_back(make_tuple(u,s,get_degree_unweighted(u-2)*alpha));
        for (itype j = ai[u-2]; j < ai[u-2+1]; j ++) {
            vtype v = aj[j]+2;
            if (R_map.count(v-2) == 0) {
                //EL.push_back(make_tuple(v,t,get_degree_unweighted(v-2)*beta));
                //EL.push_back(make_tuple(t,v,get_degree_unweighted(v-2)*beta));
                EL.push_back(make_tuple(u,v,1.0));
                EL.push_back(make_tuple(v,u,1.0));
                B_map[v];
                ABR ++;
            }
            else {
                EL.push_back(make_tuple(u,v,1.0));
                ARR ++;
            }
        }
    }
    
    cout << " ARR " << ARR << " ABR " << ABR << endl;
    
    for (auto iter = B_map.begin(); iter != B_map.end(); ++iter){
        vtype v = iter->first;
        EL.push_back(make_tuple(v,t,get_degree_unweighted(v-2)*beta));
        EL.push_back(make_tuple(t,v,get_degree_unweighted(v-2)*beta));
    }
    
    /*
    for (vtype i = 0; i < nR; i ++) {
        vtype u = R[i]+2;
        EL[make_pair(s,u)] = get_degree_unweighted(u-2)*alpha;
        for (itype j = ai[u-2]; j < ai[u-2+1]; j ++) {
            vtype v = aj[j]+2;
            EL[make_pair(u,v)] = 1;
            EL[make_pair(v,u)] = 1;
            EL[make_pair(v,t)] = get_degree_unweighted(v-2)*beta;
        }
    }
     */
}

template<typename vtype>
void init_fullyvisited_R(unordered_map<vtype,vtype>& fullyvisited, unordered_map<vtype,vtype>& R_map, vtype nR, vtype* R)
{
    for (vtype i = 0; i < nR; i ++) {
        fullyvisited[R[i]];
        R_map[R[i]];
    }
}

template<typename vtype, typename itype>
void graph<vtype,itype>::update_VL(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,
               vector<vtype>& E)
{
    vtype num = VL.size();
    vtype u,v;
    for (vtype i = 0; i < E.size(); i ++) {
        u = E[i]+2;
        if (VL.count(u) == 0) {
            VL[u] = num;
            VL_rev[num] = u;
            num ++;
        }
        for (itype j = ai[u-2]; j < ai[u-2+1]; j ++) {
            v = aj[j]+2;
            if (VL.count(v) == 0) {
                VL[v] = num;
                VL_rev[num] = v;
                num ++;
            }
        }
    }
}

template<typename vtype, typename itype>
void graph<vtype,itype>::update_EL(vector<tuple<vtype,vtype,double>>& EL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& W_map,
                                   vtype s, vtype t, double alpha, double beta)
{
    EL.clear();
    unordered_map<vtype,vtype> B_map;
    unordered_map<vtype,vtype> WnR_map;
    vtype ARR  = 0;
    vtype ABR  = 0;
    
    
    //Build ARR, AWR, AW
    for (auto iter = W_map.begin(); iter != W_map.end(); ++iter){
        vtype u = iter->first+2;
        for (itype j = ai[u-2]; j < ai[u-2+1]; j ++) {
            vtype v = aj[j]+2;
            EL.push_back(make_tuple(u,v,1.0));
        }
    }
    
    //Build WnR, B, ABR, ABW
    for (auto iter = W_map.begin(); iter != W_map.end(); ++iter){
        vtype u = iter->first+2;
        if (R_map.count(u-2) == 0) {
            WnR_map[u];
        }
        for (itype j = ai[u-2]; j < ai[u-2+1]; j ++) {
            vtype v = aj[j]+2;
            if (W_map.count(v-2) == 0) {
                B_map[v];
                EL.push_back(make_tuple(u,v,1.0));
                EL.push_back(make_tuple(v,u,1.0));
            }
        }
    }
    
    //Build sR, ARR
    for (auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first+2;
        EL.push_back(make_tuple(s,u,get_degree_unweighted(u-2)*alpha));
        EL.push_back(make_tuple(u,s,get_degree_unweighted(u-2)*alpha));
    }
    
    cout << " ARR " << ARR << " ABR " << ABR << endl;
    
    //Build tB
    for (auto iter = B_map.begin(); iter != B_map.end(); ++iter){
        vtype v = iter->first;
        EL.push_back(make_tuple(v,t,get_degree_unweighted(v-2)*beta));
        EL.push_back(make_tuple(t,v,get_degree_unweighted(v-2)*beta));
    }
    
    //Build tWnR
    for (auto iter = WnR_map.begin(); iter != WnR_map.end(); ++iter){
        vtype v = iter->first;
        EL.push_back(make_tuple(v,t,get_degree_unweighted(v-2)*beta));
        EL.push_back(make_tuple(t,v,get_degree_unweighted(v-2)*beta));
    }
    
}

template<typename vtype, typename itype>
void assemble_graph(vtype** mincut, vtype** Q, vtype** fin, vtype** pro, vtype** another_pro,
                    vtype** dist, vtype** next, vtype** to, double** flow, double** cap,
                    vtype nverts, itype nedges, vector<tuple<vtype,vtype,double>>& EL,
                    unordered_map<vtype,vtype>& VL)
{
    vtype u,v;
    double w;
    itype nEdge = 0;
    *mincut = (vtype*)malloc(sizeof(vtype) * nverts);
    *Q = (vtype*)malloc(sizeof(vtype) * nverts);
    *fin = (vtype*)malloc(sizeof(vtype) * nverts);
    *pro = (vtype*)malloc(sizeof(vtype) * nverts);
    *another_pro = (vtype*)malloc(sizeof(vtype) * nverts);
    *dist = (vtype*)malloc(sizeof(vtype) * nverts);
    *flow = (double*)malloc(sizeof(double) * 2 * nedges);
    *cap = (double*)malloc(sizeof(double) * 2 * nedges);
    *next = (vtype*)malloc(sizeof(vtype) * 2 * nedges);
    *to = (vtype*)malloc(sizeof(vtype) * 2 * nedges);
    fill(*fin, *fin + nverts, -1);
    fill(*mincut, *mincut + nverts, 0);
    for (auto iter = EL.begin(); iter != EL.end(); ++iter) {
        u = get<0>(*iter);
        v = get<1>(*iter);
        w = get<2>(*iter);
        new_edge<vtype,itype>(VL[u],VL[v],w,*to,*cap,*flow,*next,*fin,&nEdge);
    }
}

template<typename vtype>
void free_space(vtype* mincut, vtype* Q, vtype* fin, vtype* pro, vtype* another_pro,
                vtype* dist, vtype* next, vtype* to, double* flow, double* cap)
{
    free(mincut);
    free(Q);
    free(fin);
    free(pro);
    free(another_pro);
    free(dist);
    free(next);
    free(to);
    free(flow);
    free(cap);
}

template<typename vtype, typename itype>
void graph<vtype,itype>::STAGEFLOW(double delta, double alpha, double beta, unordered_map<vtype,vtype>& fullyvisited,
                                   unordered_map<vtype,vtype>& R_map)
{
    unordered_map<vtype,vtype> VL;
    unordered_map<vtype,vtype> VL_rev;
    vector<tuple<vtype,vtype,double>> EL;
    vtype s = 0;
    vtype t = 1;
    init_VL(VL,VL_rev,R_map,s,t);
    init_EL(EL,R_map,s,t,alpha,beta);
    cout << "EL size " << EL.size() << endl;
    double F = 0;
    vtype nverts = VL.size();
    itype nedges = EL.size();
    vtype *mincut = NULL, *Q = NULL, *fin = NULL, *pro = NULL, *another_pro = NULL, *dist = NULL, *next = NULL, *to = NULL;
    double *flow = NULL, *cap = NULL;
    assemble_graph<vtype,itype>(&mincut,&Q,&fin,&pro,&another_pro,&dist,&next,&to,&flow,&cap,nverts,
                                nedges,EL,VL);
    cout << "here" << endl; 
    pair<double, vtype> retData = max_flow_SL<vtype,itype>(s,t,Q,fin,pro,dist,next,to,mincut,another_pro,flow,
                                              cap,nverts);
    cout << "ok " << get<0>(retData) << " " << get<1>(retData) << endl;
    //vtype* source_set = (vtype*)malloc(sizeof(vtype) * get<1>(retData));

    vector<vtype> E;
    for (vtype i = 2; i < nverts; i ++){
        if (mincut[i] > 0 and fullyvisited.count(VL_rev[i]-2) == 0) {
            E.push_back(VL_rev[i]-2);
            fullyvisited[VL_rev[i]-2];
        }
    }
    while (E.size() > 0 && get<1>(retData) > 1) {
        update_EL(EL, R_map, fullyvisited, s, t, alpha, beta);
        update_VL(VL, VL_rev, E);
        nverts = VL.size();
        nedges = EL.size();
        free_space<vtype>(mincut,Q,fin,pro,another_pro,dist,next,to,flow,cap);
        assemble_graph<vtype,itype>(&mincut,&Q,&fin,&pro,&another_pro,&dist,&next,&to,&flow,&cap,nverts,
                                nedges,EL,VL);
        retData = max_flow_SL<vtype,itype>(s,t,Q,fin,pro,dist,next,to,mincut,another_pro,flow,cap,nverts);
        cout << "ok " << get<0>(retData) << " " << get<1>(retData) << endl;
        E.clear();
        for (vtype i = 2; i < nverts; i ++){
            if (mincut[i] > 0 and fullyvisited.count(VL_rev[i]-2) == 0) {
                E.push_back(VL_rev[i]-2);
                fullyvisited[VL_rev[i]-2];
            }
        }
    }

    free_space<vtype>(mincut,Q,fin,pro,another_pro,dist,next,to,flow,cap);
}

template<typename vtype, typename itype>
vtype graph<vtype,itype>::SimpleLocal(vtype nR, vtype* R, vtype* ret_set, double delta)
{
    unordered_map<vtype,vtype> fullyvisited;
    unordered_map<vtype,vtype> R_map;
    init_fullyvisited_R(fullyvisited, R_map, nR, R);
    pair<itype, itype> set_stats = get_stats(fullyvisited,fullyvisited.size());
    double alpha = 1.0 * get<1>(set_stats) / min(get<0>(set_stats), ai[n] - get<0>(set_stats));
    double fR = 1.0 * get<0>(set_stats) / (ai[n] - get<0>(set_stats));
    double alph0;
    double beta = alpha * (fR + delta);
    alph0 = alpha;
    cout << "here" << endl;
    STAGEFLOW(delta, alpha, beta, fullyvisited, R_map);

    cout << "here" << endl;
    set_stats = get_stats(fullyvisited,fullyvisited.size());
    alpha = 1.0 * get<1>(set_stats) / min(get<0>(set_stats), ai[n] - get<0>(set_stats));
    while (alpha < alph0) {
        alph0 = alpha;
        beta = alpha * (fR + delta);
        STAGEFLOW(delta, alpha, beta, fullyvisited, R_map);
        set_stats = get_stats(fullyvisited,fullyvisited.size());
        alpha = 1.0 * get<1>(set_stats) / min(get<0>(set_stats), ai[n] - get<0>(set_stats));
    }
    vtype actual_length = fullyvisited.size();
    vtype pos = 0;
    for (auto iter = fullyvisited.begin(); iter != fullyvisited.end(); ++ iter) {
        ret_set[pos++] = iter->first;
    }
    //cout << alpha << min(get<0>(set_stats), ai[n] - get<0>(set_stats)) << endl;

    return actual_length;
    //return 0;
}

#endif
