#ifndef ROUTINES_HPP
#define ROUTINES_HPP

#include <unordered_map>
#include "ppr_path.hpp"
#include <vector>
#include "sparseheap.hpp" // include our heap functions
#include "sparserank.hpp" // include our sorted-list functions
#include "sparsevec.hpp" // include our sparse hashtable functions

using namespace std;


template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

// Only for pairs of std::hash-able types for simplicity.
// You can of course template this struct to allow other hash functions
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        size_t seed = 0;
        hash_combine(seed,p.first);
        hash_combine(seed,p.second);
        
        //cout << p.first << " " << p.second << " " << seed << endl;
        
        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return seed;
    }
};

template<typename vtype, typename itype>
class graph{
    itype m; //number of edges
    vtype n; //number of vertices
    itype* ai; //Compressed sparse row representation
    vtype* aj; //Compressed sparse row representation
    double* a; //Compressed sparse row representation
    vtype offset; //offset for zero based arrays (matlab) or one based arrays (julia)
    double* degrees; //degrees of vertices
    double volume; //the volume of graph, 2m for undirected graph
public:
    //declare constructors
    graph<vtype,itype>(itype,vtype,itype*,vtype*,double*,vtype,double*);
    
    //common functions
    double get_degree_weighted(vtype id);
    vtype get_degree_unweighted(vtype id);
    pair<itype, itype> get_stats(unordered_map<vtype, vtype>& R_map, vtype nR);
    
    //declare routines
    //functions in aclpagerank.cpp
    vtype aclpagerank(double alpha, double eps, vtype* seedids, vtype nseedids,
                      vtype maxsteps, vtype* xids, vtype xlength, double* values);
    vtype pprgrow(double alpha, double eps, vtype* seedids, vtype nseedids,
                  vtype maxsteps, vtype* xids, vtype xlength, double* values);
    //functions in aclpagerank_weighted.cpp
    vtype aclpagerank_weighted(double alpha, double eps, vtype* seedids, vtype nseedids,
                      vtype maxsteps, vtype* xids, vtype xlength, double* values);
    vtype pprgrow_weighted(double alpha, double eps, vtype* seedids, vtype nseedids,
                  vtype maxsteps, vtype* xids, vtype xlength, double* values);
    //functions in sweepcut.cpp
    vtype sweepcut_with_sorting(double* value, vtype* ids, vtype* results,
                                vtype num, double* ret_cond);
    vtype sweepcut_without_sorting(vtype* ids, vtype* results, vtype num,
                                   double* ret_cond);
    vtype sweep_cut(vtype* ids, vtype* results, vtype num, double* ret_cond);
    //functions in ppr_path.hpp
    vtype ppr_path(double alpha, double eps, double rho, vtype* seedids, vtype nseedids, vtype* xids,
                   vtype xlength, struct path_info ret_path_results, struct rank_info ret_rank_results);
    void hypercluster_graphdiff_multiple(const vector<vtype>& set, double t, double eps, double rho,
                                         eps_info<vtype>& ep_stats, rank_record<vtype>& rkrecord, vector<vtype>& cluster);
    void graphdiffseed(sparsevec& set, const double t, const double eps_min, const double rho, const vtype max_push_count,
                       eps_info<vtype>& ep_stats, rank_record<vtype>& rkrecord, vector<vtype>& cluster);
    bool resweep(vtype r_end, vtype r_start, sparse_max_rank<vtype,double,size_t>& rankinfo, sweep_info<vtype>& swinfo);
    vtype rank_permute(vector<vtype> &cluster, vtype r_end, vtype r_start);
    void copy_array_to_index_vector(const vtype* v, vector<vtype>& vec, vtype num);
    //functions in MQI.cpp
    vtype MQI(vtype nR, vtype* R, vtype* ret_set);
    void build_map(unordered_map<vtype, vtype>& R_map,unordered_map<vtype, vtype>& degree_map,
                   vtype* R, vtype nR);
    //functions in proxl1PRaccel.cpp
    vtype proxl1PRaccel(double alpha, double rho, vtype* v, vtype v_nums, double* d,
                        double* ds, double* dsinv, double epsilon, double* grad, double* p, double* y,
                        vtype maxiter,double max_time);
    //functions in densest_subgraph.cpp
    double densest_subgraph(vtype *ret_set, vtype *actual_length);
    //functions in SimpleLocal.cpp
    void STAGEFLOW(double delta, double alpha, double beta, unordered_map<vtype,vtype>& fullyvisited, unordered_map<vtype,vtype>& R_map);
    vtype SimpleLocal(vtype nR, vtype* R, vtype* ret_set, double delta);
    void init_VL(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,unordered_map<vtype,vtype>& R_map, vtype s, vtype t);
    void init_EL(vector<tuple<vtype,vtype,double>>& EL, unordered_map<vtype,vtype>& R_map, vtype s, vtype t, double alpha, double beta);
    void update_VL(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev, vector<vtype>& E);
    void update_EL(vector<tuple<vtype,vtype,double>>& EL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& W_map,
                   vtype s, vtype t, double alpha, double beta);
};

template<typename vtype, typename itype>
graph<vtype,itype>::graph(itype _m, vtype _n, itype* _ai, vtype* _aj, double* _a,
                          vtype _offset, double* _degrees)
{
    m = _m;
    n = _n;
    ai = _ai;
    aj = _aj;
    a = _a;
    offset = _offset;
    degrees = _degrees;
    volume = (double)ai[n];
}

template<typename vtype,typename itype>
double graph<vtype,itype>::get_degree_weighted(vtype id)
{
    double d = 0;
    for(vtype j = ai[id] - offset; j < ai[id+1] - offset; j ++){
        d += a[j];
    }
    return d;
}

template<typename vtype,typename itype>
vtype graph<vtype,itype>::get_degree_unweighted(vtype id)
{
    return ai[id + 1] - ai[id];
}

template<typename vtype, typename itype>
pair<itype, itype> graph<vtype,itype>::get_stats(unordered_map<vtype, vtype>& R_map, vtype nR)
{
    itype curvol = 0;
    itype curcutsize = 0;
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++ R_iter){
        vtype v = R_iter->first;
        itype deg = get_degree_unweighted(v);
        curvol += deg;
        for(itype j = ai[v] - offset; j < ai[v + 1] - offset; j ++){
            if(R_map.count(aj[j] - offset) == 0){
                curcutsize ++;
            }
        }
    }
    
    pair<itype, itype> set_stats (curvol, curcutsize);
    return set_stats;
}

#endif
