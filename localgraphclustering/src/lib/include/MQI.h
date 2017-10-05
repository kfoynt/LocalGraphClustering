#ifndef MQI_H
#define MQI_H

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
pair<double, vtype> max_flow_MQI(itype* ai, vtype* aj, vtype offset, double a, double c, itype nedges, vtype nverts,
                             unordered_map<vtype, vtype>R_map, unordered_map<vtype, vtype>degree_map, 
                             vtype src, vtype dest, vtype* mincut, vtype* Q, vtype* fin, vtype* pro, vtype* another_pro,
                             vtype* dist, double* flow, double* cap, vtype *next, vtype *to);

template<typename vtype, typename itype>
void build_map(itype* ai, vtype* aj, vtype offset, unordered_map<vtype, vtype>& R_map, 
        unordered_map<vtype, vtype>& degree_map, vtype& R, vtype nR);


#include "../maxflow.cpp"
#include "../MQI.cpp"
#endif
