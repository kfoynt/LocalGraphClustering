#ifndef ROUTINES_HPP
#define ROUTINES_HPP

#include <unordered_map>
#include "ppr_path.hpp"
#include <vector>
#include "sparseheap.hpp" // include our heap functions
#include "sparserank.hpp" // include our sorted-list functions
#include "sparsevec.hpp" // include our sparse hashtable functions
#include <tuple>

using namespace std;

template<typename vtype,typename itype>
struct Edge
{
    vtype v ; 
    double flow ; 
    double C; 
    itype rev; 
};

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
    pair<double, double> get_stats_weighted(unordered_map<vtype, vtype>& R_map, vtype nR);
    pair<itype, itype> get_stats_rel(unordered_map<vtype, vtype>& S_map, unordered_map<vtype, vtype>& R_map, vtype nR, double delta);
    pair<double, double> get_stats_rel_weighted(unordered_map<vtype, vtype>& S_map, unordered_map<vtype, vtype>& R_map, vtype nR, double delta);
    void addEdge(vtype u, vtype v, double C);
    bool BFS(vtype s, vtype t, vtype V);
    double sendFlow(vtype u, double flow, vtype t, vector<vtype>& start, vector<pair<int,double>>& SnapShots);
    //double sendFlow(vtype u, double flow, vtype t, vector<vtype>& start);
    pair<double,vtype> DinicMaxflow(vtype s, vtype t, vtype V, vector<bool>& mincut);
    void find_cut(vtype u, vector<bool>& mincut, vtype& length);

    //common data
    int* level;
    vector< Edge<vtype,itype> > *adj;
    
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
    void build_list(unordered_map<vtype, vtype>& R_map, unordered_map<vtype, vtype>& degree_map, vtype src, vtype dest, itype a, itype c);

    
    //functions in MQI_weighted.cpp
    vtype MQI_weighted(vtype nR, vtype* R, vtype* ret_set);
    void build_map_weighted(unordered_map<vtype, vtype>& R_map,unordered_map<vtype, double>& degree_map,
                   vtype* R, vtype nR, double* degrees);
    void build_list_weighted(unordered_map<vtype, vtype>& R_map, unordered_map<vtype, double>& degree_map, vtype src, vtype dest, 
                   double a, double c, double* degrees);


    //functions in proxl1PRaccel.cpp
    vtype proxl1PRaccel(double alpha, double rho, vtype* v, vtype v_nums, double* d,
                        double* ds, double* dsinv, double epsilon, double* grad, double* p, double* y,
                        vtype maxiter,double max_time, bool use_distribution, double* distribution);
                        
    // functions in proxl1PRrand.cpp
    vtype proxl1PRrand(vtype num_nodes, vtype* seed, vtype num_seeds, double epsilon, double alpha, double rho, double* q, double* d, double* ds, double* dsinv, vtype maxiter, vtype* candidates);
//     // functions in proxl1PRrand.cpp
//     vtype proxl1PRrand_unnormalized(vtype num_nodes, vtype* seed, vtype num_seeds, double epsilon, double alpha, double rho, double* q, double* d, double* ds, double* dsinv, double* grad, vtype maxiter);
//     //functions in proxl1PRaccel.cpp
    vtype proxl1PRaccel_unnormalized(double alpha, double rho, vtype* v, vtype v_nums, double* d,
                        double* ds, double* dsinv, double epsilon, double* grad, double* p, double* y,
                        vtype maxiter, double max_time, bool use_distribution, double* distribution);


    //functions in densest_subgraph.cpp
    double densest_subgraph(vtype *ret_set, vtype *actual_length);
    void build_list_DS(double g, vtype src, vtype dest);


    //functions in SimpleLocal.cpp
    void STAGEFLOW(double delta, double alpha, double beta, unordered_map<vtype,vtype>& fullyvisited, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& S);
    vtype SimpleLocal(vtype nR, vtype* R, vtype* ret_set, double delta, bool relcondflag);
    void init_VL(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,unordered_map<vtype,vtype>& R_map);
    void init_EL(vector< tuple<vtype,vtype,double> >& EL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& VL, vtype s, vtype t, double alpha, double beta);
    void update_VL(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev, vector<vtype>& E);
    void update_EL(vector< tuple<vtype,vtype,double> >& EL, unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& W_map,
                   vtype s, vtype t, double alpha, double beta);
    void assemble_graph(vector<bool>& mincut, vtype nverts, itype nedges, vector<tuple<vtype,vtype,double>>& EL);

    //functions in SimpleLocal_weighted.cpp
    void STAGEFLOW_weighted(double delta, double alpha, double beta, unordered_map<vtype,vtype>& fullyvisited, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& S);
    vtype SimpleLocal_weighted(vtype nR, vtype* R, vtype* ret_set, double delta, bool relcondflag);
    void init_VL_weighted(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev,unordered_map<vtype,vtype>& R_map);
    void init_EL_weighted(vector< tuple<vtype,vtype,double> >& EL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& VL, vtype s, vtype t, double alpha, double beta);
    void update_VL_weighted(unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& VL_rev, vector<vtype>& E);
    void update_EL_weighted(vector< tuple<vtype,vtype,double> >& EL, unordered_map<vtype,vtype>& VL, unordered_map<vtype,vtype>& R_map, unordered_map<vtype,vtype>& W_map,
                   vtype s, vtype t, double alpha, double beta);
    void assemble_graph_weighted(vector<bool>& mincut, vtype nverts, itype nedges, vector<tuple<vtype,vtype,double>>& EL);
    

    //functions for capacity releasing diffusion
    vtype capacity_releasing_diffusion(vector<vtype>& ref_node, vtype U,vtype h,vtype w,vtype iterations,vtype* cut);
    void unit_flow(unordered_map<vtype,double>& Delta, vtype U, vtype h, vtype w, unordered_map<vtype,double>& f_v, 
        unordered_map<vtype,double>& ex, unordered_map<vtype,vtype>& l);
    void round_unit_flow(unordered_map<vtype,vtype>& l, unordered_map<vtype,double>& cond,unordered_map<vtype,vector<vtype>>& labels);

    //functions in triangleclusters.cpp
    void triangleclusters(double* cond, double* cut, double* vol, double* cc, double* t);
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
    volume = 0;
    if (ai != NULL) {
        volume = (double)ai[n];
    }
    adj = NULL;
    level = NULL;

}

template<typename vtype, typename itype>
void graph<vtype,itype>::addEdge(vtype u, vtype v, double C)
{
    // Forward edge : 0 flow and C capacity
    Edge<vtype,itype> p{v, 0, C, (itype)adj[v].size()};
 
    // Back edge : 0 flow and C capacity
    Edge<vtype,itype> q{u, 0, C, (itype)adj[u].size()};
 
    adj[u].push_back(p);
    adj[v].push_back(q); // reverse edge
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

template<typename vtype, typename itype>
pair<double, double> graph<vtype,itype>::get_stats_weighted(unordered_map<vtype, vtype>& R_map, vtype nR)
{
    double curvol = 0;
    double curcutsize = 0;
    for(auto R_iter = R_map.begin(); R_iter != R_map.end(); ++ R_iter){
        vtype v = R_iter->first;
        double deg = degrees[v];
        curvol += deg;
        for(itype j = ai[v] - offset; j < ai[v + 1] - offset; j ++){
            if(R_map.count(aj[j] - offset) == 0){
                curcutsize += a[j];
            }
        }
    }
    
    pair<double, double> set_stats (curvol, curcutsize);
    return set_stats;
}

// This function and the following function return the same cut as "get_stats" or "get_stats_weighted" but relative volume values between set S and seeds set R
// vol(S,R) = vol(S,R) - delta*vol(S,Rc)
template<typename vtype, typename itype>
pair<itype, itype> graph<vtype,itype>::get_stats_rel(unordered_map<vtype, vtype>& S_map, unordered_map<vtype, vtype>& R_map, vtype nR, double delta)
{
    itype curvol = 0;
    itype curcutsize = 0;
    itype deg;
    for(auto S_iter = S_map.begin(); S_iter != S_map.end(); ++ S_iter){
        vtype v = S_iter->first;
        if (R_map.count(v) != 0) {
            deg = get_degree_unweighted(v);
            curvol += deg;
        }
        else {
            deg = get_degree_unweighted(v);
            curvol -= delta*deg;
        }
        for(itype j = ai[v] - offset; j < ai[v + 1] - offset; j ++){
            if(S_map.count(aj[j] - offset) == 0){
                curcutsize ++;
            }
        }
    }
    
    pair<itype, itype> set_stats (curvol, curcutsize);
    return set_stats;
}


template<typename vtype, typename itype>
pair<double, double> graph<vtype,itype>::get_stats_rel_weighted(unordered_map<vtype, vtype>& S_map, unordered_map<vtype, vtype>& R_map, vtype nR, double delta)
{
    double curvol = 0;
    double curcutsize = 0;
    double deg;
    for(auto S_iter = S_map.begin(); S_iter != S_map.end(); ++ S_iter){
        vtype v = S_iter->first;
        if (R_map.count(v) != 0) {
            deg = degrees[v];
            curvol += deg;
        }
        else {
            deg = degrees[v];
            curvol -= delta*deg;
        }
        for(itype j = ai[v] - offset; j < ai[v + 1] - offset; j ++){
            if(S_map.count(aj[j] - offset) == 0){
                curcutsize += a[j];
            }
        }
    }
    
    pair<double, double> set_stats (curvol, curcutsize);
    return set_stats;
}

#include "maxflow.cpp"

#endif
