/**
 * INPUT:
 *     alpha     - teleportation parameter between 0 and 1
 *     rho       - l1-reg. parameter
 *     v         - seed node
 *     ai,aj,a   - Compressed sparse row representation of A
 *     d         - vector of node strengths
 *     epsilon   - accuracy for termination criterion
 *     n         - size of A
 *     ds        - the square root of d
 *     dsinv     - 1/ds
 *     offset    - offset for zero based arrays (matlab) or one based arrays (julia)
 *
 * OUTPUT:
 *     p              - PageRank vector as a row vector
 *     not_converged  - flag indicating that maxiter has been reached
 *     grad           - last gradient
 *
 */

#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <time.h>
#include <vector>
#include <unordered_set>

#include "include/proxl1PRaccel_c_interface.h"
#include "include/routines.hpp"

using namespace std;

#define INDEX(idx, offset) (idx - offset)

template<typename vtype>
struct FistaParams {
    vtype num_iter;
    vtype max_iter;

    double start_runtime;
    double max_runtime;
    double alpha;
    double epsilon;
    double rho;

    vector<double> x;
    vector<double> y;
    vector<double> gradient;
    unordered_set<vtype> nonzero_set;
};

template<typename vtype, typename itype>
struct GraphParams {
    double* a;
    itype* ai;
    vtype* aj;

    double* d;
    double* ds;
    double* dsinv;
};

///////////////////////
// Function declaration
///////////////////////

template<typename vtype, typename itype>
double get_dx(vtype node, const FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params);

template<typename vtype, typename itype>
unordered_set<vtype> update_gradient(vector<pair<vtype, double> >& node_to_dy, FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params);

template<typename vtype, typename itype>
void update_nonzeros(unordered_set<vtype>& nodes_touched, FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params);

template<typename vtype, typename itype>
bool is_nonzero(vtype node, const FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params);

template<typename vtype, typename itype>
bool is_converged(const FistaParams<vtype>& fista_params, GraphParams<vtype, itype>& graph_params);

//////////////////////
// Function definition
//////////////////////

template<typename vtype, typename itype>
void update(FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params) {
    double beta = (1 - sqrt(fista_params.alpha)) / (1 + sqrt(fista_params.alpha));
    vector<pair<vtype, double> > node_to_dy;

    for (vtype node : fista_params.nonzero_set) {
        double dx = get_dx(node, fista_params, graph_params);
        fista_params.x[node] += dx;
        double dy = fista_params.x[node] + beta * dx - fista_params.y[node];
        fista_params.y[node] += dy;
        node_to_dy.push_back(make_pair(node, dy));
    }

    unordered_set<vtype> nodes_touched = update_gradient(node_to_dy, fista_params, graph_params);
    update_nonzeros(nodes_touched, fista_params, graph_params);
}

template<typename vtype, typename itype>
double get_dx(vtype node, const FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params) {
    double threshold = fista_params.alpha * fista_params.rho * graph_params.ds[node];
    double dx;

    if (fista_params.y[node] - fista_params.gradient[node] > threshold) {
        dx = fista_params.y[node] - fista_params.gradient[node] - threshold - fista_params.x[node];
    } else if (fista_params.y[node] - fista_params.gradient[node] < -threshold) {
        dx = fista_params.y[node] - fista_params.gradient[node] + threshold - fista_params.x[node];
    } else {
        dx = -fista_params.x[node];
    }
    return dx;
}

template<typename vtype, typename itype>
unordered_set<vtype> update_gradient(vector<pair<vtype, double> >& node_to_dy, FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params) {
    unordered_set<vtype> nodes_touched;

    for (auto iter : node_to_dy) {
        vtype node = iter.first;
        double dy = iter.second;
        fista_params.gradient[node] += .5 * (1 + fista_params.alpha) * dy;
        nodes_touched.insert(node);

        for (itype neighbor_idx = graph_params.ai[node]; neighbor_idx < graph_params.ai[node + 1]; ++neighbor_idx) {
            vtype neighbor = graph_params.aj[neighbor_idx];
            fista_params.gradient[neighbor] -= .5 * (1 - fista_params.alpha) * graph_params.dsinv[node] * graph_params.dsinv[neighbor] * graph_params.a[neighbor_idx] * dy;
            nodes_touched.insert(neighbor);
        }
    }

    return nodes_touched;
}

template<typename vtype, typename itype>
void update_nonzeros(unordered_set<vtype>& nodes_touched, FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params) {
    for (vtype node : nodes_touched) {
        if (is_nonzero(node, fista_params, graph_params)) {
            fista_params.nonzero_set.insert(node);
        } else {
            fista_params.nonzero_set.erase(node);
        }
    }
}

template<typename vtype, typename itype>
bool is_nonzero(vtype node, const FistaParams<vtype>& fista_params, const GraphParams<vtype, itype>& graph_params) {
    double FLOAT_PRECISION = 1e-9;
    double threshold = fista_params.rho * fista_params.alpha * graph_params.ds[node];

    if (fabs(fista_params.x[node]) > FLOAT_PRECISION || fabs(fista_params.y[node]) > FLOAT_PRECISION) {
        return true;
    } else if (fista_params.gradient[node] < -threshold || fista_params.gradient[node] > threshold) {
        return true;
    }
    return false;
}

template<typename vtype, typename itype>
bool is_converged(FistaParams<vtype>& fista_params, GraphParams<vtype, itype>& graph_params) {
    if (++fista_params.num_iter >= fista_params.max_iter || double(clock()) - fista_params.start_runtime >= fista_params.max_runtime) {
        return true;
    }

    for (vtype node : fista_params.nonzero_set) {
        if (fabs(fista_params.gradient[node]) > (1 + fista_params.epsilon) * fista_params.rho * fista_params.alpha * graph_params.ds[node]) {
            return false;
        }
    }
    return true;
}


template<typename vtype, typename itype>
vtype graph<vtype,itype>::proxl1PRaccel(double alpha, double rho, vtype* v, vtype v_nums, double* d,
                                        double* ds, double* dsinv, double epsilon, double* grad, double* p,
                                        double* y, vtype maxiter, double max_time, bool use_distribution, double* distribution)
{
    GraphParams<vtype, itype> graph_params = {this->a, this->ai, this->aj, d, ds, dsinv};
    FistaParams<vtype> fista_params = {
        0,
        maxiter,
        double(clock()),
        max_time * 1000,
        alpha,
        epsilon,
        rho,
        vector<double>(this->n, 0),
        vector<double>(this->n, 0),
        vector<double>(this->n, 0),
        unordered_set<vtype>()
    };

    if (use_distribution) {
        for (vtype idx = 0; idx < this->n; ++idx) {
            fista_params.nonzero_set.insert(idx);
            fista_params.gradient[idx] = -fista_params.alpha * graph_params.dsinv[idx] * distribution[idx];
        }
    } else {
        for (vtype idx = 0; idx < v_nums; ++idx) {
            vtype seed_node = v[idx];
            fista_params.nonzero_set.insert(seed_node);
            fista_params.gradient[seed_node] = -fista_params.alpha * graph_params.dsinv[seed_node] / v_nums;
        }
    }
    while (!is_converged(fista_params, graph_params)) {
        update(fista_params, graph_params);
    }

    for (vtype node = 0; node < this->n; ++node) {
        p[node] = fabs(fista_params.x[node]) * ds[node];
    }
    
    return 0;
}

template<typename vtype, typename itype>
vtype graph<vtype,itype>::proxl1PRaccel_unnormalized(double alpha, double rho, vtype* v, vtype v_nums, double* d,
                                        double* ds, double* dsinv, double epsilon, double* grad, double* p,
                                        double* y, vtype maxiter,double max_time, bool use_distribution, double* distribution)
{
    // TODO: implement unnormalized method
    return this->proxl1PRaccel(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,y,maxiter,max_time, use_distribution, distribution);
}

uint32_t proxl1PRaccel32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double alpha,
                         double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                         double* dsinv, double epsilon, double* grad, double* p, double* y,
                         uint32_t maxiter, uint32_t offset,double max_time, bool normalized_objective, bool use_distribution, double* distribution)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRaccel(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,y,maxiter,max_time, use_distribution, distribution);
    }
    else{
        return g.proxl1PRaccel_unnormalized(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,y,maxiter,max_time, use_distribution, distribution);
    }
}

int64_t proxl1PRaccel64(int64_t n, int64_t* ai, int64_t* aj, double* a, double alpha,
                        double rho, int64_t* v, int64_t v_nums, double* d, double* ds,
                        double* dsinv,double epsilon, double* grad, double* p, double* y,
                        int64_t maxiter, int64_t offset,double max_time,bool normalized_objective, bool use_distribution,  double* distribution)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRaccel(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,y,maxiter,max_time, use_distribution, distribution);
    }
    else{
        return g.proxl1PRaccel_unnormalized(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,y,maxiter,max_time, use_distribution, distribution);
    }
}

uint32_t proxl1PRaccel32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, double alpha,
                            double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                            double* dsinv, double epsilon, double* grad, double* p, double* y,
                            uint32_t maxiter, uint32_t offset,double max_time,bool normalized_objective, bool use_distribution,  double* distribution)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    if (normalized_objective){
        return g.proxl1PRaccel(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,y,maxiter,max_time, use_distribution, distribution);
    }
    else{
        return g.proxl1PRaccel_unnormalized(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,y,maxiter,max_time, use_distribution, distribution);
    }
}