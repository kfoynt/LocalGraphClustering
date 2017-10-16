#ifndef DENSEST_SUBGRAPH_H
#define DENSEST_SUBGRAPH_H

#include <utility>

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
pair<double, vtype> max_flow_ds(itype* ai, vtype* aj, vtype offset, double a, double* degrees, vtype n,
                                itype m, vtype src, vtype dest, vtype *Q, vtype *fin, vtype *pro,
                                vtype *dist, vtype *next, vtype *to, vtype *cut,
                                vtype *another_pro, vtype *pro3, double *flow, double *cap, double g);

#include "../maxflow.cpp"
#include "../densest_subgraph.cpp"
#endif
