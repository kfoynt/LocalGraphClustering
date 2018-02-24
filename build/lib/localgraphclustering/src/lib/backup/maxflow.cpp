#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <utility>
#include <typeinfo>

using namespace std;

template<typename vtype, typename itype>
void new_edge(vtype u, vtype v, double weight, vtype *to, double *cap, double *flow, vtype *next, vtype *fin, itype *nEdge)
{
    to[*nEdge] = v;
    cap[*nEdge] = weight;
    flow[*nEdge] = 0;
    next[*nEdge] = fin[u];
    fin[u] = (*nEdge) ++;
    to[*nEdge] = u;
    cap[*nEdge] = weight;
    flow[*nEdge] = weight;
    next[*nEdge] = fin[v];
    fin[v] = (*nEdge) ++;
}

template<typename vtype, typename itype>
bool dinic_bfs(vtype nverts, vtype src, vtype dest, vtype *dist, vtype *Q, vtype *fin, vtype *next, vtype *to, double *flow, double *cap) 
{
    vtype st, en, i, u, v;
    fill(dist, dist + nverts, -1);
    dist[src] = st = en = 0;
    Q[en ++] = src;
    while(st < en) 
    {
        u = Q[st ++];
        for(i = fin[u]; i >= 0; i = next[i]) 
        {
            v = to[i];
            if(flow[i] < cap[i] && dist[v] == -1) 
            {
                dist[v] = dist[u] +1;
                Q[en ++] = v;
            }
        }
    }
    return dist[dest] != -1;
}

template<typename vtype, typename itype>
double dinic_dfs(vtype u, double fl, vtype src, vtype dest, vtype *pro, vtype *next, vtype *to, vtype *dist, double *cap, double *flow) 
{
    if(u == dest) return fl;
    vtype v;
    double df;
    for(vtype &e=pro[u]; e >= 0; e = next[e]) 
    {
        v = to[e];
        if(flow[e] < cap[e] && dist[v] == dist[u] + 1) 
        {
            if(u == src || (cap[e] - flow[e]) <= fl)
            {
                fl = cap[e] - flow[e];
            }
            df = dinic_dfs<vtype, itype>(v, fl, src, dest, pro, next, to, dist, cap, flow);
            if(df>0) 
            {
                flow[e] += df;
                flow[e^1] -= df;
                return df;
            }
        }
    }
    return 0;
}

template<typename vtype, typename itype>
void find_cut(vtype u, vtype *cut, vtype *another_pro, vtype *next, vtype *to, double *flow, double *cap, vtype* length)
{
    cut[u] = 1;
    *length = *length + 1;
    for(vtype &e = another_pro[u]; e >= 0; e = next[e]){
        vtype v = to[e];
        if(flow[e] < cap[e] && cut[v] == 0){
            find_cut<vtype, itype>(v, cut, another_pro, next, to, flow, cap, length);
        }
    }
}


template<typename vtype, typename itype>
pair<double, vtype> max_flow_MQI(itype* ai, vtype* aj, vtype offset, double a, double c, itype nedges, vtype nverts,
                             unordered_map<vtype, vtype>R_map, unordered_map<vtype, vtype>degree_map, 
                             vtype src, vtype dest, vtype* mincut, vtype* Q, vtype* fin, vtype* pro, vtype* another_pro,
                             vtype* dist, double* flow, double* cap, vtype *next, vtype *to)
{
    //cout << "nverts " << nverts << " nedges " << nedges << endl;

    fill(fin, fin + nverts, -1);
    fill(mincut, mincut + nverts, 0);
    itype nEdge = 0;
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
        vtype u = R_iter->first;
        vtype u1 = R_iter->second;
        for(vtype j = ai[u] - offset; j < ai[u + 1] - offset; j ++){
            vtype v = aj[j] - offset;
            auto got = R_map.find(v);
            if(R_map.count(v) > 0){
                vtype v1 = got->second;
                double w = a;
                new_edge<vtype, itype>(u1, v1, w, to, cap, flow, next, fin, &nEdge);
            }
        }
    }
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++R_iter){
        vtype u1 = src;
        vtype v = R_iter->first;
        vtype v1 = R_iter->second;
        auto got = degree_map.find(v);
        double w = a * got->second;
        new_edge<vtype,itype>(u1, v1, w, to, cap, flow, next, fin, &nEdge);
        u1 = v1;
        v1 = dest;
        vtype u = v;
        w = c * (ai[u + 1] - ai[u]);
        new_edge<vtype,itype>(u1, v1, w, to, cap, flow, next, fin, &nEdge);
    }

    double ret = 0;
    double df;
    while(dinic_bfs<vtype, itype>(nverts, src, dest, dist, Q, fin, next, to, flow, cap)) 
    {
        for(vtype i = 0; i < nverts; i++) 
        {
            pro[i] = fin[i];
            another_pro[i] = fin[i];
        }
        while(true) 
        {
            df = dinic_dfs<vtype, itype>(src, 0, src, dest, pro, next, to, dist, cap, flow);
            if(df) ret += df;
            else break;
        }
    }
    vtype length = 0;
    find_cut<vtype, itype>(src, mincut, another_pro, next, to, flow, cap, &length);
    pair<double, vtype> retData (ret,length);
   
    return retData;
}

template<typename vtype, typename itype>
double max_flow_ds(itype* ai, vtype* aj, double* a, double* degrees, vtype n,
                                itype m, vtype src, vtype dest, vtype *Q, vtype *fin, vtype *pro,
                                vtype *dist, vtype *next, vtype *to, vtype *cut,
                                vtype *another_pro, vtype *pro3, double *flow, double *cap, double g)
{
    
    vtype nverts = n + 2;
    fill(fin, fin + nverts, -1);
    fill(cut, cut + nverts, 0);
    itype nEdge = 0;
    //cout << "nverts " << nverts << endl;
    for(size_t i = 0; i < (size_t)n; i ++){
        //cout << i << " " << ai[i] << " " << ai[i+1] << endl;
        for(size_t j = ai[i]; j < (size_t)ai[i+1]; j ++){
            //cout << "aj= " << aj[j] << endl;
            //cout << "a= " << a[j] << endl;
            new_edge<vtype,itype>(i+1,aj[j]+1,a[j],to,cap,flow,next,fin,&nEdge);
        }
    }
    //cout << "start" << endl;
    for(size_t i = 0; i < (size_t)n; i ++){
        new_edge<vtype,itype>(src,i+1,m*1.0/2,to,cap,flow,next,fin,&nEdge);
    }
    //cout << "start" << endl;
    for(size_t i = 0; i < (size_t)n; i ++){
        new_edge<vtype,itype>(i+1,dest,m*1.0/2+2*g-degrees[i],to,cap,flow,next,fin,&nEdge);
    }
    
    double ret = 0;
    double df;
    //cout << "start" << endl;
    while(dinic_bfs<vtype, itype>(nverts, src, dest, dist, Q, fin, next, to, flow, cap))
    {
        for(vtype i = 0; i < nverts; i++)
        {
            pro[i] = fin[i];
            another_pro[i] = fin[i];
            pro3[i] = fin[i];
        }
        while(true)
        {
            df = dinic_dfs<vtype, itype>(src, 0, src, dest, pro, next, to, dist, cap, flow);
            if(df) ret += df;
            else break;
        }
    }
    vtype length = 0;
    find_cut<vtype, itype>(src, cut, another_pro, next, to, flow, cap, &length);
    (void)length;
    
    return ret;
}


template<typename vtype, typename itype>
pair<double, vtype> max_flow_SL(vtype src, vtype dest, vtype* Q, vtype* fin, vtype* pro, vtype* dist, vtype *next, vtype *to,
                                 vtype* mincut, vtype* another_pro, double* flow, double* cap, vtype nverts)
{
    //cout << "nverts " << nverts << " nedges " << nedges << endl;
    double ret = 0;
    double df;
    while(dinic_bfs<vtype, itype>(nverts, src, dest, dist, Q, fin, next, to, flow, cap))
    {
        for(vtype i = 0; i < nverts; i++)
        {
            pro[i] = fin[i];
            another_pro[i] = fin[i];
        }
        while(true)
        {
            df = dinic_dfs<vtype, itype>(src, 0, src, dest, pro, next, to, dist, cap, flow);
            if(df) ret += df;
            else break;
        }
    }
    vtype length = 0;
    find_cut<vtype, itype>(src, mincut, another_pro, next, to, flow, cap, &length);
    pair<double, vtype> retData (ret,length);
    
    return retData;
}
