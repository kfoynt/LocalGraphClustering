#ifndef SIMPLELOCAL_H
#define SIMPLELOCAL_H

#include <utility>
#include <unordered_map>

using namespace std;

template<typename vtype, typename itype>
void new_edge(vtype u, vtype v, double weight, vtype *to, double *cap, double *flow, vtype *next, vtype *fin, itype *nEdge);

template<typename vtype, typename itype>
bool dinic_bfs(vtype nverts, vtype src, vtype dest, vtype *dist, vtype *Q, vtype *fin, vtype *next, vtype *to, double *flow, double *cap);

template<typename vtype, typename itype>
double dinic_dfs(vtype u, double fl, vtype src, vtype dest, vtype *pro, vtype *next, vtype *to, vtype *dist, double *cap, double *flow);

template<typename vtype, typename itype>
void find_cut(vtype u, vtype *cut, vtype *another_pro, vtype *next, vtype *to, double *flow, double *cap, vtype* length);

template<typename vtype, typename itype>
pair<double, vtype> max_flow_SL(vtype src, vtype dest, vtype* Q, vtype* fin, vtype* pro, vtype* dist, vtype *next, vtype *to,
                                vtype* mincut, vtype* another_pro, double* flow, double* cap);


#include "../maxflow.cpp"
#include "../SimpleLocal.cpp"
#endif
