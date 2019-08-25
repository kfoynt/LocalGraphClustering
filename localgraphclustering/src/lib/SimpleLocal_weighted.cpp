#ifdef SIMPLELOCAL_WEIGHTED_H

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <stdint.h>
#include <typeinfo>
#include <functional>
#include "include/routines.hpp"
#include "include/SimpleLocal_weighted_c_interface.h"
#include <tuple>
#include <fstream>
#include <limits>
#include <numeric> 


using namespace std;

template<typename vtype, typename itype>
void graph<vtype,itype>::init_VL_weighted(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,
                                 unordered_map<vtype,vtype>& R_map)
{
    //cout << "changed" << endl;
    vtype num = 1;
    for (auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first;
        if (VL.count(u) == 0) {
            VL[u] = num;
            VL_rev[num] = u;
            num ++;
        }
        for (itype j = ai[u]; j < ai[u+1]; j ++) {
            vtype v = aj[j];
            if (VL.count(v) == 0) {
                VL[v] = num;
                VL_rev[num] = v;
                num ++;
            }
        }
    }
}

template<typename vtype, typename itype>
void graph<vtype,itype>::init_EL_weighted(vector<tuple<vtype,vtype,double>>& EL, unordered_map<vtype,vtype>& R_map,
                                 unordered_map<vtype,vtype>& VL, vtype s, vtype t, double alpha, double beta)
{
    //cout << "alpha " << alpha << " beta " << beta << endl;
    //unordered_map<vtype,vtype> R_map;
    unordered_map<vtype,vtype> B_map;
    vtype ARR  = 0;
    vtype ABR  = 0;
    for (auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first;
        EL.push_back(make_tuple(s,VL[u],get_degree_weighted(u)*alpha));
        EL.push_back(make_tuple(VL[u],s,get_degree_weighted(u)*alpha));
        for (itype j = ai[u]; j < ai[u+1]; j ++) {
            vtype v = aj[j];
            if (R_map.count(v) == 0) {
                //EL.push_back(make_tuple(v,t,get_degree_unweighted(v-2)*beta));
                //EL.push_back(make_tuple(t,v,get_degree_unweighted(v-2)*beta));
                EL.push_back(make_tuple(VL[u],VL[v],a[j]));
                EL.push_back(make_tuple(VL[v],VL[u],a[j]));
                B_map[v];
                ABR ++;
            }
            else {
                EL.push_back(make_tuple(VL[u],VL[v],a[j]));
                ARR ++;
            }
        }
    }

    //cout << " ARR " << ARR << " ABR " << ABR << endl;

    for (auto iter = B_map.begin(); iter != B_map.end(); ++iter){
        vtype v = iter->first;
        EL.push_back(make_tuple(VL[v],t,get_degree_weighted(v)*beta));
        EL.push_back(make_tuple(t,VL[v],get_degree_weighted(v)*beta));
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
void graph<vtype,itype>::update_VL_weighted(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,
               vector<vtype>& E)
{
    vtype num = VL.size()+1;
    vtype u,v;
    for (vtype i = 0; i < E.size(); i ++) {
        u = E[i];
        if (VL.count(u) == 0) {
            VL[u] = num;
            VL_rev[num] = u;
            num ++;
        }
        for (itype j = ai[u]; j < ai[u+1]; j ++) {
            v = aj[j];
            if (VL.count(v) == 0) {
                VL[v] = num;
                VL_rev[num] = v;
                num ++;
            }
        }
    }
}

template<typename vtype, typename itype>
void graph<vtype,itype>::update_EL_weighted(vector<tuple<vtype,vtype,double>>& EL, unordered_map<vtype,vtype>& VL,
                                   unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& W_map,
                                   vtype s, vtype t, double alpha, double beta)
{
    if (!EL.empty() && EL.size() > 0) {
        EL.clear();
    }

    unordered_map<vtype,vtype> B_map;
    unordered_map<vtype,vtype> WnR_map;
    vtype ARR  = 0;
    vtype ABR  = 0;


    //Build ARR, AWR, AW
    for (auto iter = W_map.begin(); iter != W_map.end(); ++iter){
        vtype u = iter->first;
        for (itype j = ai[u]; j < ai[u+1]; j ++) {
            vtype v = aj[j];
            if (W_map.count(v) > 0) {
                EL.push_back(make_tuple(VL[u],VL[v],a[j]));
            }
        }
    }

    //Build WnR, B, ABR, ABW
    for (auto iter = W_map.begin(); iter != W_map.end(); ++iter){
        vtype u = iter->first;
        if (R_map.count(u) == 0) {
            WnR_map[u];
        }
        for (itype j = ai[u]; j < ai[u+1]; j ++) {
            vtype v = aj[j];
            if (W_map.count(v) == 0) {
                B_map[v];
                EL.push_back(make_tuple(VL[u],VL[v],a[j]));
                EL.push_back(make_tuple(VL[v],VL[u],a[j]));
            }
        }
    }

    //Build sR, ARR
    //cout << "for now: " << EL.size() << endl;
    for (auto iter = R_map.begin(); iter != R_map.end(); ++iter){
        vtype u = iter->first;
        EL.push_back(make_tuple(s,VL[u],get_degree_weighted(u)*alpha));
        EL.push_back(make_tuple(VL[u],s,get_degree_weighted(u)*alpha));
    }

    //cout << "ARR " << ARR << " ABR " << ABR << endl;

    //Build tB
    for (auto iter = B_map.begin(); iter != B_map.end(); ++iter){
        vtype v = iter->first;
        EL.push_back(make_tuple(VL[v],t,get_degree_weighted(v)*beta));
        EL.push_back(make_tuple(t,VL[v],get_degree_weighted(v)*beta));
    }

    //Build tWnR
    for (auto iter = WnR_map.begin(); iter != WnR_map.end(); ++iter){
        vtype v = iter->first;
        EL.push_back(make_tuple(VL[v],t,get_degree_weighted(v)*beta));
        EL.push_back(make_tuple(t,VL[v],get_degree_weighted(v)*beta));
    }

}

template<typename vtype, typename itype>
void graph<vtype,itype>::assemble_graph_weighted(vector<bool>& mincut, vtype nverts, itype nedges,
                                        vector<tuple<vtype,vtype,double>>& EL)
{
    vtype u,v;
    double w;
    mincut.resize(nverts);

    for (auto iter = EL.begin(); iter != EL.end(); ++iter) {
        u = get<0>(*iter);
        v = get<1>(*iter);
        w = get<2>(*iter);

        addEdge(u,v,w);
    }
}

template<typename vtype, typename itype>
void free_space(int* level, vector< Edge<vtype,itype> > *adj)
{
    delete[] adj;
    delete[] level;
}


template<typename vtype, typename itype>
void save_EL(vector<tuple<vtype,vtype,double>>& EL)
{
    ofstream wptr;
    wptr.open("EL.smat", std::ofstream::out | std::ofstream::trunc);
    vtype u,v;
    double w;
    for (auto iter = EL.begin(); iter != EL.end(); ++iter) {
        u = get<0>(*iter);
        v = get<1>(*iter);
        w = get<2>(*iter);
        wptr << u << " " << v << " " << w << endl;
    }
    wptr.close();
}


template<typename vtype, typename itype>
void graph<vtype,itype>::STAGEFLOW_weighted(double delta, double alpha, double beta, unordered_map<vtype,vtype>& fullyvisited,
                                   unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& S)
{
    //cout << "begin STAGEFLOW" << endl;
    unordered_map<vtype,vtype> VL;
    unordered_map<vtype,vtype> VL_rev;
    vector<tuple<vtype,vtype,double>> EL;
    vtype s = 0;
    init_VL_weighted(VL,VL_rev,R_map);
    vtype t = VL.size()+1;
    init_EL_weighted(EL,R_map,VL,s,t,alpha,beta);
    //cout << "EL size " << EL.size() << endl;

    //double F = 0;
    vtype nverts = VL.size()+2;
    itype nedges = EL.size();


    adj = new vector<Edge<vtype,itype>>[nverts];
    level = new int[nverts];
    vector<bool> mincut;
    assemble_graph(mincut,nverts,nedges,EL);
    itype sum = 0;
    for (vtype i = 0; i < nverts; i ++) {
        sum += adj[i].size();
    }
    //cout << "here sum: " << sum << endl;


    /*
    pair<double, vtype> retData = max_flow_SL<vtype,itype>(s,t,Q,fin,pro,dist,next,to,mincut,another_pro,flow,
                                              cap,nverts);
    */
    pair<double, vtype> retData = DinicMaxflow(s,t,nverts,mincut);



    //cout << "ok " << get<0>(retData) << " " << get<1>(retData) << endl;
    //vtype* source_set = (vtype*)malloc(sizeof(vtype) * get<1>(retData));


    //TODO
    vector<vtype> E;
    for (vtype i = 1; i < nverts-1; i ++){
        if (mincut[i] && fullyvisited.count(VL_rev[i]) == 0) {
            E.push_back(VL_rev[i]);
            fullyvisited[VL_rev[i]];
        }
    }
    free_space<vtype,itype>(level, adj);


    while (E.size() > 0 && get<1>(retData) > 1) {
        update_VL_weighted(VL, VL_rev, E);
        t = VL.size()+1;
        update_EL_weighted(EL, VL, R_map, fullyvisited, s, t, alpha, beta);
        //cout << "EL size " << EL.size() << " VL size " << VL.size() << endl;

        nverts = VL.size()+2;
        nedges = EL.size();
        //free_space<vtype,itype>(level, adj);
        adj = new vector<Edge<vtype,itype>>[nverts];
        level = new int[nverts];
        assemble_graph(mincut,nverts,nedges,EL);
        sum = 0;
        for (vtype i = 0; i < nverts; i ++) {
            sum += adj[i].size();
        }

        //retData = max_flow_SL<vtype,itype>(s,t,Q,fin,pro,dist,next,to,mincut,another_pro,flow,cap,nverts);
        retData = DinicMaxflow(s,t,nverts,mincut);

        //cout << "ok " << get<0>(retData) << " " << get<1>(retData) << endl;
        if (!E.empty() && E.size() > 0) {
            E.clear();
        }
        for (vtype i = 1; i < nverts-1; i ++){
            if (mincut[i] && fullyvisited.count(VL_rev[i]) == 0) {
                E.push_back(VL_rev[i]);
                fullyvisited[VL_rev[i]];
            }
        }
        free_space<vtype,itype>(level, adj);

    }

    for (vtype i = 1; i < nverts-1; i ++){
        if (mincut[i]) {
            S[VL_rev[i]];
        }
    }

    //free_space<vtype,itype>(level, adj);
}

template<typename vtype, typename itype>
void clear_map(unordered_map<vtype,itype>& M)
{
    if (!M.empty() && M.size() > 0) {
        M.clear();
    }
}

template<typename vtype, typename itype>
void copy_results(unordered_map<vtype,vtype>& S, vtype* ret_set, vtype* actual_length)
{
	*actual_length = S.size();
	vtype pos = 0;
    for (auto iter = S.begin(); iter != S.end(); ++ iter) {
         ret_set[pos++] = iter->first;
    }
}

template<typename vtype, typename itype>
vtype graph<vtype,itype>::SimpleLocal_weighted(vtype nR, vtype* R, vtype* ret_set, double delta, bool relcondflag)
{
    unordered_map<vtype,vtype> fullyvisited, S;
    unordered_map<vtype,vtype> R_map;
    vtype actual_length;

    init_fullyvisited_R(fullyvisited, R_map, nR, R);
    double total_degree = std::accumulate(degrees,degrees+n,0);
    pair<double, double> set_stats;

    set_stats = get_stats_weighted(fullyvisited,fullyvisited.size());
    double fR = 1.0 * get<0>(set_stats) / (total_degree - get<0>(set_stats));
    double alpha;
    if (relcondflag == true) {
        set_stats = get_stats_rel_weighted(fullyvisited,R_map,R_map.size(),fR + delta);
        if ((get<0>(set_stats) <= 0) || (fullyvisited.size() == 0) || (fullyvisited.size() == n)) {
            alpha = numeric_limits<double>::max();
        }
        else {
            alpha = 1.0 * get<1>(set_stats) / get<0>(set_stats);
        }
        
    }
    else {
        if (min(get<0>(set_stats), total_degree - get<0>(set_stats)) != 0)
        {
            alpha = 1.0 * get<1>(set_stats) / min(get<0>(set_stats), total_degree - get<0>(set_stats));
        }
        else {
            alpha = numeric_limits<double>::max();
        }
    }
    //cout << alpha << " " << get<0>(set_stats) << " " << get<1>(set_stats) << endl;

    double alph0;
    double beta = alpha * (fR + delta);
    alph0 = alpha;

    clear_map<vtype,vtype>(S);
    STAGEFLOW_weighted(delta, alpha, beta, fullyvisited, R_map, S);

    // if (S.size() == 0) {
    // 	copy_results<vtype,itype>(R_map,ret_set,&actual_length);
    // 	return actual_length;
    // }
    if (relcondflag == true) {
        set_stats = get_stats_rel_weighted(S,R_map,R_map.size(),fR + delta);
        if ((get<0>(set_stats) <= 0) || (S.size() == 0) || (S.size() == n)) {
            alpha = numeric_limits<double>::max();
        }
        else {
            alpha = 1.0 * get<1>(set_stats) / get<0>(set_stats);
        }
    }
    else {
        set_stats = get_stats_weighted(S,S.size());
        if (min(get<0>(set_stats), total_degree - get<0>(set_stats)) != 0)
        {
            alpha = 1.0 * get<1>(set_stats) / min(get<0>(set_stats), total_degree - get<0>(set_stats));
        }
        else {
            alpha = numeric_limits<double>::max();
        }
    }
    //cout << alpha << " " << get<0>(set_stats) << " " << get<1>(set_stats) << endl;
    
    //cout << "after first step: " << alpha << endl;
    if (alpha >= alph0) {
        copy_results<vtype,itype>(R_map,ret_set,&actual_length);
        return actual_length;
    }
    while (alpha < alph0) {
        //cout << alpha << endl;
        copy_results<vtype,itype>(S,ret_set,&actual_length);
        alph0 = alpha;
        beta = alpha * (fR + delta);
        clear_map<vtype,vtype>(fullyvisited);
        clear_map<vtype,vtype>(R_map);
        init_fullyvisited_R(fullyvisited, R_map, nR, R);
        clear_map<vtype,vtype>(S);
        STAGEFLOW_weighted(delta, alpha, beta, fullyvisited, R_map, S);
        if (relcondflag == true) {
            set_stats = get_stats_rel_weighted(S,R_map,R_map.size(),fR + delta);
            if ((get<0>(set_stats) <= 0) || (S.size() == 0) || (S.size() == n)) {
                alpha = numeric_limits<double>::max();
            }
            else {
                alpha = 1.0 * get<1>(set_stats) / get<0>(set_stats);
            }
        }
        else {
            set_stats = get_stats_weighted(S,S.size());
            if (min(get<0>(set_stats), total_degree - get<0>(set_stats)) != 0)
            {
                alpha = 1.0 * get<1>(set_stats) / min(get<0>(set_stats), total_degree - get<0>(set_stats));
            }
            else {
                alpha = numeric_limits<double>::max();
            }
        }
        //cout << alpha << " " << get<0>(set_stats) << " " << get<1>(set_stats) << endl;
    }

    //cout << alpha << min(get<0>(set_stats), ai[n] - get<0>(set_stats)) << endl;
    return actual_length;
    //return 0;
}

#endif
