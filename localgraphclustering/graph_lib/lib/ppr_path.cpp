/**
 * Implement a seeded ppr clustering scheme that finds
 * the best cluster for all tolerances eps in an interval.
 * Returns information about solution vector, residual,
 * and best cluster at every push step, and every
 * new value of epsilon reached.
 *
 * INPUT:
 *     n        - the number of vertices in the graph
 *     ai,aj    - Compressed sparse row representation
 *     offset   - offset for zero based arrays (matlab) or one based arrays (julia)
 *     alpha    - value of alpha
 *     eps      - value of epsilon
 *     rho      - value of rho
 *     seedids  - the set of indices for seeds
 *     nseedids - the number of indices in the seeds
 *     maxsteps - the max number of steps
 *     xlength  - the max number of ids in the solution vector
 *     xids     - the solution vector, i.e. the vertices with nonzero pagerank value
 *     values   - the pagerank value vector for xids (already sorted in decreasing order)
 *     ret_path_results - the path results under every different epsilon
 *     ret_rank_results - the rank results after each step of queue update
 *
 * OUTPUT:
 *     actual_length - the number of nonzero entries in the solution vector
 *
 * COMPILE:
 *     make ppr_path
 *
 * EXAMPLE:
 *     Use functions from readData.hpp to read a graph and seed from files.
 *     double alpha = 0.99;
 *     double eps = 0.0001;
 *     double rho = 0.1;
 *     int64_t max_step = (int64_t)1 / ((1 - alpha) * eps);
 *     int64_t* xids = (int64_t*)malloc(sizeof(int64_t)*m);
 *     int64_t num_eps = 0;
 *     double* epsilon = (double*)malloc(sizeof(double) * max_step);
 *     double* conds = (double*)malloc(sizeof(double) * max_step);
 *     double* cuts = (double*)malloc(sizeof(double) * max_step);
 *     double* vols = (double*)malloc(sizeof(double) * max_step);
 *     int64_t* setsizes = (int64_t*)malloc(sizeof(int64_t) * max_step);
 *     int64_t* stepnums = (int64_t*)malloc(sizeof(int64_t) * max_step);
 *     struct path_info ret_path_results = {.num_eps = &num_eps, .epsilon = epsilon,
 *     .conds = conds, .cuts = cuts, .vols = vols, .setsizes = setsizes, .stepnums = stepnums};
 *
 *     int64_t nrank_changes = 0, nrank_inserts = 0, nsteps = 0, size_for_best_cond = 0;
 *     int64_t* starts = (int64_t*)malloc(sizeof(int64_t) * max_step);
 *     int64_t* ends = (int64_t*)malloc(sizeof(int64_t) * max_step);
 *     int64_t* nodes = (int64_t*)malloc(sizeof(int64_t) * max_step);
 *     int64_t* deg_of_pushed = (int64_t*)malloc(sizeof(int64_t) * max_step);
 *     int64_t* size_of_solvec = (int64_t*)malloc(sizeof(int64_t) * max_step);
 *     int64_t* size_of_r = (int64_t*)malloc(sizeof(int64_t) * max_step);
 *     double* val_of_push = (double*)malloc(sizeof(double) * max_step);
 *     double* global_bcond = (double*)malloc(sizeof(double) * max_step);
 *     struct rank_info ret_rank_results = {.starts = starts, .ends = ends, .nodes = nodes,
 *     .deg_of_pushed = deg_of_pushed, .size_of_solvec = size_of_solvec, .size_of_r = size_of_r,
 *     .val_of_push = val_of_push, .global_bcond = global_bcond, .nrank_changes = &nrank_changes,
 *     .nrank_inserts = &nrank_inserts, .nsteps = &nsteps, .size_for_best_cond = &size_for_best_cond};
 *
 *     int64_t actual_length = ppr_path64(m,ai,aj,0,alpha,eps,rho,seedids,nseedids,xids,m,
 *     ret_path_results,ret_rank_results);
 */

#include "include/ppr_path_c_interface.h"

#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <algorithm>
#include <stdint.h>
#include <assert.h>
#include <vector>
#include <math.h>

#include "include/sparsehash/dense_hash_map.h"

#include "include/routines.hpp"

using namespace std;

/*
 * @param n, ai, aj, offset - Compressed sparse row representation, 
 *                            with offset for zero based (matlab) or 
 *                            one based arrays (julia)
 * @param alpha - value of alpha
 * @param eps - value of epsilon
 * @param rho - value of rho
 * @param seedids - indices of seed set
 * @param nseedids - number of indices in seed set
 * @param xids, xlength, values - the solution vector
 * */


template<typename vtype, typename itype>
struct rank_map{
    typedef google::dense_hash_map<vtype, vtype> Map;
};


template<typename vtype, typename itype>
vtype graph<vtype,itype>::ppr_path(double alpha, double eps, double rho, vtype* seedids, vtype nseedids, vtype* xids,
                                   vtype xlength, struct path_info ret_path_results, struct rank_info ret_rank_results)
{
    cout << "preprocessing start: " << endl;

    vector<vtype> seedset;
    copy_array_to_index_vector(seedids, seedset, nseedids);
    cout << "preprocessing end: " << endl;

    eps_info<vtype> eps_stats(1000);
    rank_record<vtype> rkrecord;
    vector<vtype> bestclus;

    cout << "call to hypercluster_graphdiff() start" << endl;

    hypercluster_graphdiff_multiple(seedset, alpha, eps, rho, eps_stats, rkrecord, bestclus);

    cout << "call to hypercluster_graphdiff() DONE" << endl;

    xlength = bestclus.size();

    for(size_t i = 0; i < xlength; ++ i){
        xids[i] = bestclus[i] + offset;
    }
    *(ret_path_results.num_eps) = eps_stats.num_epsilons;
    for(size_t i = 0; i < eps_stats.num_epsilons; i ++){
        ret_path_results.epsilon[i] = eps_stats.epsilons[i];
        ret_path_results.conds[i] = eps_stats.conds[i];
        ret_path_results.cuts[i] = eps_stats.cuts[i];
        ret_path_results.vols[i] = eps_stats.vols[i];
        ret_path_results.setsizes[i] = eps_stats.setsizes[i];
        ret_path_results.stepnums[i] = eps_stats.stepnums[i];
    }
    
    *(ret_rank_results.nrank_changes) = rkrecord.nrank_changes;
    *(ret_rank_results.nrank_inserts) = rkrecord.nrank_inserts;
    *(ret_rank_results.nsteps) = rkrecord.nsteps;
    *(ret_rank_results.size_for_best_cond) = rkrecord.size_for_best_cond;
    for(size_t i = 0; i < rkrecord.nsteps; i ++){
        ret_rank_results.starts[i] = rkrecord.starts[i];
        ret_rank_results.ends[i] = rkrecord.ends[i];
        ret_rank_results.nodes[i] = rkrecord.nodes[i];
        ret_rank_results.deg_of_pushed[i] = rkrecord.deg_of_pushed[i];
        ret_rank_results.size_of_solvec[i] = rkrecord.size_of_solvec[i];
        ret_rank_results.size_of_r[i] = rkrecord.size_of_r[i];
        ret_rank_results.val_of_push[i] = rkrecord.val_of_push[i];
        ret_rank_results.global_bcond[i] = rkrecord.global_bcond[i];
    }
    return xlength;
}

/*
 * returns index of the best conductance in the vector swinfo.conductances
 */
template<typename vtype, typename itype>
bool graph<vtype,itype>::resweep(vtype r_end, vtype r_start, sparse_max_rank<vtype,double,size_t>& rankinfo,
                                 sweep_info<vtype>& swinfo)
{

    // ensure sweep_info vectors are big enough
    if ( r_start >= swinfo.cut.size() ){
        swinfo.cut.resize((r_start+1)*2);
        swinfo.vol.resize((r_start+1)*2);
        swinfo.cond.resize((r_start+1)*2);
        swinfo.rank_of_best_cond.resize((r_start+1)*2);
    }
    double old_bcond = swinfo.best_cond_global;
    (swinfo.num_sweeps) += (r_start-r_end+1);
    double total_degree = volume;
    bool was_there_new_bcond = 0;
    vtype rank_of_best_cond = 0;

//  FAST WAY/*
    vtype gindex = rankinfo.rank_to_index(r_end);
    double deg = get_degree_unweighted(gindex);
    std::vector<double> neighbors_ranked_less_than(r_start-r_end+1,0.0);
    std::vector<double> oldcut(r_start-r_end+1,0.0);
    std::vector<double> oldvol(r_start-r_end+1,0.0);

    // get rankings of neighbors of the shifted node
    for (vtype nzi = ai[gindex] - offset; nzi < ai[gindex+1] - offset; nzi++){
        vtype temp_gindex = aj[nzi] - offset;
        vtype temp_rank = rankinfo.index_to_rank(temp_gindex);
        if ( (temp_rank < r_end) && (temp_rank >= 0) ){ neighbors_ranked_less_than[0] += 1.0; }
        if ( (temp_rank > r_end) && (temp_rank <= r_start) ){
            neighbors_ranked_less_than[temp_rank-r_end] = 1.0;
        }
    }
    for (vtype j = 1; j <= (r_start-r_end); j++){
        neighbors_ranked_less_than[j] += neighbors_ranked_less_than[j-1];
    }
    
    // get old cut/vol information
    if (r_end == 0){
        oldcut[0] = 0.0;
        oldvol[0] = 0.0;
    }
    else{
        oldcut[0] = swinfo.cut[r_end-1];
        oldvol[0] = swinfo.vol[r_end-1];
        rank_of_best_cond = swinfo.rank_of_best_cond[r_end-1];
    }    
    for (vtype j = 1; j <= (r_start-r_end); j++){
        oldcut[j] = swinfo.cut[r_end-1+j];
        oldvol[j] = swinfo.vol[r_end-1+j];
    }

    // update volumes and cuts from r_end to r_start
    double cur_cond = 1.0;
    for (vtype j = 0; j <= (r_start-r_end); j++){
        double cur_vol = oldvol[j] + deg;
        swinfo.vol[r_end+j] = cur_vol;
        double cur_cut = oldcut[j] + deg - 2.0*neighbors_ranked_less_than[j];
        swinfo.cut[r_end+j] = cur_cut;
        
        if (cur_vol == 0.0 || cur_vol == total_degree) { cur_cond = 1.0; }
        else { cur_cond = cur_cut/std::min(cur_vol,total_degree-cur_vol); }
    }

    // finally, compute conductance values from r_end to r_start
    for (vtype cur_rank = r_end; cur_rank <= r_start; cur_rank++){
        double cur_cut = swinfo.cut[cur_rank];
        double cur_vol = swinfo.vol[cur_rank];
        if (cur_vol == 0.0 || cur_vol == total_degree) { cur_cond = 1.0; }
        else { cur_cond = cur_cut/std::min(cur_vol,total_degree-cur_vol); }
        swinfo.cond[cur_rank] = cur_cond;
    }

    // ... and update 'rank_of_best_cond' for all indices r_end to r_start
    if ( r_start > swinfo.back_ind ) { swinfo.back_ind = r_start; }
    for (vtype cur_rank = r_end; cur_rank <= swinfo.back_ind; cur_rank++){
        if ( swinfo.cond[cur_rank] < swinfo.cond[rank_of_best_cond] ){ rank_of_best_cond = cur_rank; }
        swinfo.rank_of_best_cond[cur_rank] = rank_of_best_cond;
    }
    rank_of_best_cond = swinfo.rank_of_best_cond[swinfo.back_ind];
    cur_cond = swinfo.cond[rank_of_best_cond];
    swinfo.best_cond_this_sweep = cur_cond;


    if (cur_cond < old_bcond){ // if current best_cond_this_sweep improves...
        swinfo.rank_of_bcond_global = rank_of_best_cond;
        swinfo.best_cond_global = cur_cond; // update best_cond_global
        swinfo.vol_of_bcond_global = swinfo.vol[rank_of_best_cond];
        swinfo.cut_of_bcond_global = swinfo.cut[rank_of_best_cond];
        was_there_new_bcond = 1; // signal that a new best_cond_global was found
    }
    return was_there_new_bcond;
} // END resweep()

           

/*
 * reorders an ordered list to reflect an update to the rankings
 */
template<typename vtype, typename itype>
vtype graph<vtype,itype>::rank_permute(std::vector<vtype> &cluster, vtype r_end, vtype r_start)
{
    vtype temp_val = cluster[r_start];
    for (vtype ind = r_start; ind > r_end; ind--){ cluster[ind] = cluster[ind-1]; }
    cluster[r_end] = temp_val;
    return (r_start-r_end);
}



/**
 *  graphdiffseed inputs:
 *      G   -   adjacency matrix of an undirected graph
 *      set -   seed vector: the indices of a seed set of vertices
 *              around which cluster forms; normalized so
 *                  set[i] = 1/set.size(); )
 *  output:
 *      p = f(tP) * set
 *              with infinity-norm accuracy of eps * f(t)
 *              in the degree weighted norm
 *  parameters:
 *      t   - the value of t
 *      eps - the accuracy
 *      max_push_count - the total number of steps to run
 */
template<typename vtype, typename itype>
void graph<vtype,itype>::graphdiffseed(sparsevec& set, const double t, const double eps_min, const double rho,
                                       const vtype max_push_count, eps_info<vtype>& ep_stats, rank_record<vtype>& rkrecord,
                                       std::vector<vtype>& cluster )
{
    cout << "ppr_all_mex::graphdiffseed()  BEGIN " << endl;
    vtype npush = 0;
    vtype nsteps = 0;
    double best_eps = 1.0;
    double cur_eps = 1.0;
    std::vector<double> epsilons;
    std::vector<double> conds;
//    vtype stagnant = 0;
    
    // ***** initialize residual, solution, and bookkeeping vectors
    sparse_max_heap<vtype,double,size_t> r(1000);
    sparse_max_rank<vtype,double,size_t> solvec(1000);
    for (sparsevec::map_type::iterator it=set.map.begin(),itend=set.map.end();
         it!=itend;++it) {
        r.update(it->first,it->second); // "update" handles the heap internally
    }
    sweep_info<vtype> spstats(1000);
    
    cur_eps = r.look_max();

    cout << "ppr_all_mex::graphdiffseed()  variables declared, begin WHILE loop " << endl;

    while ( (npush < max_push_count) && (cur_eps > eps_min) ) {
        // STEP 1: pop top element off of heap
        double rij, rij_temp, rij_res;
        vtype ri = r.extractmax(rij_temp); // heap handles sorting internally
        double degofi = get_degree_unweighted(ri);
        rij_res = cur_eps*rho;
        r.update(ri, rij_res ); // handles the heap internally
        rij = rij_temp - rij_res;


        // STEP 2: update soln vector
        bool new_bcond = 0;            
        size_t rank_start;
        size_t old_size = solvec.hsize;
        size_t rank_end = solvec.update(ri, rij, rank_start); // handles sorting internally.
                // Sets rank_start to the rank ri had before it was updated.
                // Sets rank_end to the rank ri has after it was updated.

        // STEP 3: update sweeps for new solution vector
        if ( rank_start == old_size ){ // CASE (1): new entry
            new_bcond = resweep(rank_end, old_size, solvec, spstats);
            rkrecord.update_record(old_size, rank_end, ri, degofi, r.hsize, rij, spstats.best_cond_global);            
        }
        else if( (rank_start < old_size) && (rank_start > rank_end) ){ // CASE (2): existing entry changes rank
            new_bcond = resweep(rank_end, rank_start, solvec, spstats);
            rkrecord.update_record(rank_start, rank_end, ri, degofi, r.hsize, rij, spstats.best_cond_global);
        } 
        else {
            // CASE (3): no changes to sweep info, just resweep anyway.
            new_bcond = resweep(rank_end, rank_start, solvec, spstats);
            rkrecord.update_record(rank_start, rank_end, ri, degofi, r.hsize, rij, spstats.best_cond_global);
        }

        // STEP 4: update residual
        double update = t*rij;
        for (vtype nzi=ai[ri] - offset; nzi < ai[ri+1] - offset; ++nzi) {
            vtype v = aj[nzi] - offset;
            //cout << "update " << update << " sr_degree " << sr_degree(G,v) << endl;
            r.update(v,update/get_degree_unweighted(v)); // handles the heap internally
        }

        // STEP 5: update cut-set stats, check for convergence
        double cur_max = r.look_max();
        if (cur_max < cur_eps){ // we've reached a lower value of || ||_{inf},
            cur_eps = cur_max;  // so update cut stats for new val of cur_eps
            vtype rank_of_bcond = spstats.rank_of_best_cond[spstats.back_ind];            
            double loc_bcond = spstats.cond[rank_of_bcond];
            double cur_vol = spstats.vol[rank_of_bcond];
            double cur_cut = spstats.cut[rank_of_bcond];
            ep_stats.update(nsteps, cur_eps, loc_bcond,cur_cut, cur_vol, (rank_of_bcond+1) );
        }         
        if ( new_bcond == 1 ){ // new best_cond_global, so update
            best_eps = cur_eps;
            rkrecord.size_for_best_cond = rkrecord.nrank_changes;
        }
        nsteps++;        
        npush+=degofi;
    }//END 'while'
    cout << "WHILE done " << endl;
    
    //reconstruct bestcluster from the record of rank changes, rkrecord
    cluster.resize(rkrecord.nrank_inserts);
    vtype cluster_length = 0;
    vtype num_rank_swaps = 0;
    for (vtype j = 0; j < rkrecord.size_for_best_cond ; j++){
        vtype rs = rkrecord.starts[j];
        vtype re = rkrecord.ends[j];
        vtype rn = rkrecord.nodes[j];
        if (rs == rkrecord.size_of_solvec[j]){ // node rn added for the first time
            cluster[cluster_length] = rn;
            rs = cluster_length;
            cluster_length++;
        }
        num_rank_swaps += rank_permute(cluster, re, rs);
    }    
    cluster.resize(spstats.rank_of_bcond_global+1); // delete nodes outside best cluster

}  // END graphdiffseed()



/** Cluster will contain a list of all the vertices in the cluster
 * @param set - the set of starting vertices to use
 * @param t - scaling factor in f(t*A)
 * @param eps - the solution tolerance eps
 * @param p - the solution vector
 * @param r - the residual vector
 * @param a - vector which supports .push_back to add vertices for the cluster
 * @param stats - a structure for statistics of the computation
 */

template<typename vtype, typename itype>
void graph<vtype,itype>::hypercluster_graphdiff_multiple(const std::vector<vtype>& set, double t, double eps,
                                                         double rho, eps_info<vtype>& ep_stats, rank_record<vtype>& rkrecord,
                                                         std::vector<vtype>& cluster)
{
    // reset data
    sparsevec r; r.map.clear();
    cout << "beginning of hypercluster_graphdiff_multiple() " << endl;
    //size_t maxdeg = 0;
    for (size_t i=0; i<set.size(); ++i) { //populate r with indices of "set"
        assert(set[i] >= 0); assert(set[i] < n); // assert that "set" contains indices i: 1<=i<=n
        double setideg = get_degree_unweighted(set[i]);
        r.map[set[i]] += 1.0/(double)(set.size()*setideg);
        // r is normalized to be stochastic, then degree-normalized
        cout << "i = "<< i << "\t set[i] = " << set[i] << "\t setideg = " << setideg << endl;
        //maxdeg = std::max(maxdeg, setideg);
    }
    printf("at last, graphdiffseed: t=%f eps=%f \n", t, eps);
    
    const vtype max_npush = std::min( 
        (vtype)std::numeric_limits<int>::max() , (vtype)(1/((1-t)*eps)) );
    graphdiffseed(r, t, eps, rho, max_npush, ep_stats, rkrecord, cluster);

}  // END hyper_cluster_graphdiff_multiple()
            
template<typename vtype, typename itype>
void graph<vtype,itype>::copy_array_to_index_vector(const vtype* v, std::vector<vtype>& vec, vtype num)
{
    vec.resize(num);
    
    for (size_t i=0; i<num; ++i) {
        vec[i] = v[i] - offset;
    }
}  // END copy_array_to_index_vector()




int64_t ppr_path64(int64_t n, int64_t* ai, int64_t* aj, int64_t offset, double alpha,
                   double eps, double rho, int64_t* seedids, int64_t nseedids, int64_t* xids,
                   int64_t xlength, struct path_info ret_path_results, struct rank_info ret_rank_results)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.ppr_path(alpha, eps, rho, seedids, nseedids, xids, xlength, ret_path_results, ret_rank_results);
}
uint32_t ppr_path32(uint32_t n, uint32_t* ai, uint32_t* aj, uint32_t offset, double alpha,
                    double eps, double rho, uint32_t* seedids, uint32_t nseedids, uint32_t* xids,
                    uint32_t xlength, struct path_info ret_path_results, struct rank_info ret_rank_results)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.ppr_path(alpha, eps, rho, seedids, nseedids, xids, xlength, ret_path_results, ret_rank_results);
}
uint32_t ppr_path32_64(uint32_t n, int64_t* ai, uint32_t* aj, uint32_t offset, double alpha,
                       double eps, double rho, uint32_t* seedids, uint32_t nseedids, uint32_t* xids,
                       uint32_t xlength, struct path_info ret_path_results, struct rank_info ret_rank_results)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.ppr_path(alpha, eps, rho, seedids, nseedids, xids, xlength, ret_path_results, ret_rank_results);
}
