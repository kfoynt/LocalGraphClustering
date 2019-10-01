/**
 * Randomized proximal coordinate descent for l1 regularized pagerand vector
 * INPUT:
 *     alpha     - teleportation parameter between 0 and 1
 *     rho       - l1-reg. parameter
 *     v         - Seed node
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


// include write data to files
#include <iostream>
#include <fstream>
// end
#include <vector>
#include <cmath>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include "include/routines.hpp"
#include "include/proxl1PRrand_c_interface.h"
#include <random>
#include <cmath>

using namespace std;

namespace proxl1PRrand 
{
    bool fileExist(const string& fname) {
        ifstream file(fname);
        return file.good();
    }


    template<typename vtype>
    double compute_l2_norm(double* vec, vtype n){
        double l2_norm = 0;
        for(vtype i = 0; i < n; i ++){
            //cout << max_num << " " << grad[i]/ds[i] << endl;
            l2_norm += (vec[i])*(vec[i]);
        }
        l2_norm = sqrt(l2_norm);

        return l2_norm;
    }

    template<typename vtype, typename itype>
    void writeQdiag(vtype num_nodes, const string& fname, itype* ai, vtype* aj, double* d, double alpha) {
        if (fileExist(fname)) {
            return;
        }
        vector<bool> Adiag(num_nodes, false);

        for (vtype node = 0; node < num_nodes; ++node) {
            itype idx = ai[node];
            while (idx < ai[node + 1] && aj[idx] < node) ++idx;
            Adiag[node] = idx < ai[node + 1] && aj[idx] == node;
        }

        double c = (1 - alpha) / 2;
        ofstream file(fname);
        if (file.is_open()) {
            for (vtype node = 0; node < num_nodes; ++node) {
                file << (1.0 - c - c * Adiag[node] / d[node]);
                if (node < num_nodes - 1) file << ',';
            }
            file.close();
        }
    }

    template<typename vtype>
    void writeLog(vtype num_nodes, const string& fname, double* q) {
        ofstream file(fname, fstream::app);
        if (file.is_open()) {
            for (vtype i = 0; i < num_nodes; ++i) {
                file << q[i];
                if (i < num_nodes - 1) file << ',';
            }
            file << '\n';
            file.close();
        }
    }

    template<typename vtype, typename itype>
    void writeGraph(vtype num_nodes, const string& fname, itype* ai, vtype* aj) {
        if (fileExist(fname)) {
            return;
        }
        ofstream file(fname);
        if (file.is_open()) {
            itype idx = 0;
            for (vtype node = 0; node < num_nodes; ++node) {
                for (vtype neighbor = 0; neighbor < num_nodes; ++neighbor) {
                    if (idx < ai[node + 1] && aj[idx] == neighbor) {
                        ++idx;
                        file << 1;
                    } else {
                        file << 0;
                    }
                    if (neighbor < num_nodes - 1) file << ',';
                }
                file << '\n';
            }
            file.close();
        }
    }

    void writeTime(clock_t& timeStamp, const string& fname) {
        clock_t currTime = clock();
        ofstream file(fname, fstream::app);
        if (file.is_open()){
            file << double(currTime - timeStamp) / CLOCKS_PER_SEC << '\n';
            file.close();
        }
        timeStamp = currTime;
    }

    static long long g_seed = 1;

//     inline
//     long long getRand() {
//         // g_seed = (214013*g_seed+2531011); 
//         // return (g_seed>>16)&0x7FFF; 
//         random_device rd;
//         mt19937_64 e2(rd());
//         uniform_int_distribution<long long int> dist(std::llround(std::pow(2,0)), std::llround(std::pow(2,63)));
//         return dist(e2);
//     }

//     template<typename vtype, typename itype>
//     void inline updateGrad(vtype& node, double& stepSize, double& c, double& ra, unordered_map<vtype,double>* q, unordered_map<vtype,double>* grad, double* ds, double* dsinv, itype* ai, vtype* aj, double* a, double& stepszra, queue<vtype>* Q, double & threshold) {
//         double dqs = -grad->at(node)-ds[node]*ra;
//         double dq = dqs*stepSize;
//         double cdqdsinv = c*dq*dsinv[node];   
//         q->at(node) += dq;
//         grad->at(node) += dqs;
//         double grad_old;
        
//         vtype neighbor;
//         typename unordered_map<vtype,double>::const_iterator got;
        
//         for (itype j = ai[node]; j < ai[node + 1]; ++j) {
//             neighbor = aj[j];
            
//             got = grad->find(neighbor);
//             if (got == grad->end()) {
// //                 (*grad)[neighbor] = cdqdsinv*dsinv[neighbor]*a[j]; 
//                 (*grad)[neighbor] = cdqdsinv*dsinv[neighbor]; 
//                 if (grad->at(neighbor) < -threshold*ds[neighbor]) {
//                     Q->push(neighbor);
//                     (*q)[neighbor] = 0;
//                 }
//             }
//             else {
// //                 grad->at(neighbor) -= cdqdsinv*dsinv[neighbor]*a[j];
//                 grad_old = grad->at(neighbor);
//                 grad->at(neighbor) -= cdqdsinv*dsinv[neighbor];
//                 if ((grad->at(neighbor) < -threshold*ds[neighbor]) && (grad_old >= -threshold*ds[neighbor])) {
//                     Q->push(neighbor);
//                     got = q->find(neighbor);
//                     if (got == q->end()) (*q)[neighbor] = 0;
//                 }
//             }
//         }
//     }


//     template<typename vtype, typename itype>
//     void updateGrad_unnormalized(vtype& node, double& stepSize_constant, double& c, double& ra, double* q, double* grad, double* d, itype* ai, vtype* aj, double* a, bool* visited, vtype* candidates, vtype& candidates_size) {
//         double dqs = -grad[node]/d[node]-ra;
//         double dq = dqs*stepSize_constant;
//         double cdq = c*dq;
//         double stepszra = stepSize_constant*ra;
//         q[node] += dq;
//         grad[node] += dqs*d[node];

//         vtype neighbor;
//         for (itype j = ai[node]; j < ai[node + 1]; ++j) {
//             neighbor = aj[j];
//             grad[neighbor] -= cdq*a[j]; //(1 + alpha)
//             if (!visited[neighbor] && q[neighbor] - stepSize_constant*(grad[neighbor]/d[neighbor]) >= stepszra) {
//                 visited[neighbor] = true;
//                 candidates[candidates_size++] = neighbor;
//             }
//         }
//     }
    
//     template<typename vtype, typename itype>
//     void warm_start(vtype& node, double& stepSize, double& alpha, double* y, double* ds, double* dsinv, double* grad, itype* ai, vtype* aj, bool* visited, vtype* candidates, vtype& candidates_size, double& ra, double& c) {
//         grad[node] += y[node]/stepSize;
// //         cout << "grad[" << node << "]: " << grad[node] << endl;
// //         vtype neighbor;
// //         for (itype j = ai[node]; j < ai[node + 1]; ++j) {
// //             neighbor = aj[j];
// //             grad[node] -= (1-alpha)/2*dsinv[neighbor]*dsinv[node]*y[neighbor];
// //         }
        
//         vtype neighbor;
//         for (itype j = ai[node]; j < ai[node + 1]; ++j) {
        
//             neighbor = aj[j];
// //             cout << "before grad[" << node << "]: " << grad[node] << endl;
// //             cout << "before dsinv[" << neighbor << "]: " << dsinv[neighbor] << endl;
// //             cout << "before dsinv[" << node << "]: " << dsinv[node] << endl;
// //             cout << "before y[" << node << "]: " << y[node] << endl;
//             grad[node] -= c*dsinv[neighbor]*dsinv[node]*y[neighbor];
// //             cout << "after grad[" << node << "]: " << grad[node] << endl;
            
//             if (!visited[neighbor] && y[neighbor] - stepSize*grad[neighbor] >= stepSize*ds[neighbor]*ra) {
//                 visited[neighbor] = true;
//                 candidates[candidates_size++] = neighbor;
//             }
//         }
        
//     }
    
    
//     template<typename vtype, typename itype>
//     void warm_start_unnormalized(vtype& node, double& stepSize, double& alpha, double* y, double* d, double* grad, itype* ai, vtype* aj, bool* visited, vtype* candidates, vtype& candidates_size, double& ra, double& c) {
//         grad[node] += y[node]*d[node]/stepSize;
        
//         vtype neighbor;
//         for (itype j = ai[node]; j < ai[node + 1]; ++j) {
        
//             neighbor = aj[j];
//             grad[neighbor] -= c*y[node];
            
//             if (!visited[neighbor] && y[neighbor] - stepSize*(grad[neighbor]/d[neighbor]) >= stepSize*ra) {
//                 visited[neighbor] = true;
//                 candidates[candidates_size++] = neighbor;
//             }
//         }
        
//     }
}

template<typename vtype, typename itype>
vtype graph<vtype,itype>::proxl1PRrand(vtype num_nodes, vtype* seed, vtype num_seeds, double epsilon, double alpha, double rho, double* q, double* d, double* ds, double* dsinv, vtype maxiter, vtype* xids)
{
//     clock_t timeStamp1;
//     clock_t timeStamp2;
    
//     double sum_grad = 0;
//     double sum_term = 0;
//     double sum_random = 0;
    
//     timeStamp1 = clock();
//     random_device rd;
//     mt19937_64 e2(rd());
//     uniform_int_distribution<long long int> dist(std::llround(std::pow(2,0)), std::llround(std::pow(2,63)));
//     return dist(e2);
    
//     timeStamp2 = clock();
    
//     cout << "time initialization distribution: " << (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC << endl;
    
    double dqs, dq, cdqdsinv, grad_old, temp, temp2; 
    typename unordered_map<vtype, double>::const_iterator got;
    
    vtype neighbor;
    
//     timeStamp1 = clock();
    vtype node;
// 	vtype not_converged = 0;
//     vtype* candidates = new vtype[num_nodes];
//     bool* visited = new bool[num_nodes];
//     for (vtype i = 0; i < num_nodes; ++i) visited[i] = false;
//     bool* visited[ num_nodes ] = { false };
    
//     timeStamp2 = clock();
    
//     cout << "time initialization visited: " << (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC << endl;

//     timeStamp1 = clock();
    
    // initialize seed nodes as candidates
//     double maxNorm = 0;
    unordered_map<vtype,double> grad;
    unordered_map<vtype,double> values;
    queue<vtype> Q;
    
    for (vtype i = 0; i < num_seeds; ++i) {
        // set gradient and update max norm
        grad[seed[i]] = -alpha*dsinv[seed[i]]/num_seeds;
//         maxNorm = max(maxNorm, abs(grad[seed[i]]*dsinv[seed[i]]));
        // set as candidate nodes
        values[seed[i]] = 0;
        Q.push(seed[i]);
//         cout << "seed[" << i << "]: " << seed[i] << endl;
    }
    
//     double sum_ = 0;
    
//     for (pair<vtype, double> element : grad)
//     {
//         sum_ += -ds[element.first]*element.second/alpha;
//     }
    
//     for (pair<vtype, double> element : values)
//     {
//         sum_ += ds[element.first]*element.second;
//     }
    
//     cout << "sum: " << sum_ << endl;
    
//     timeStamp2 = clock();
    
//     cout << "time initialization seed nodes: " << (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC << endl;
    
//     for (vtype i = 0; i < num_nodes; ++i) {
//         cout << "grad[" << i << "]: " << grad[i] << endl;
//     }
    
    double c = (1-alpha)/2;
    double ra = rho*alpha;
    double stepSize = 2.0/(1+alpha);
    double stepszra = stepSize*ra;
    
//     // for warm start frame work
//     for (vtype i = 0; i < num_nodes; ++i) {
// //         cout << "y[" << i << "]: " << y[i] << endl;
// //         cout << "dsinv[" << i << "]: " << dsinv[i] << endl;
// //         cout << "visited[" << i << "]: " << visited[i] << endl;
//         if (y[i] != 0 && dsinv[i] != 0) {
//             //cout << "y[" << i << "]: " << y[i] << endl;
//             candidates[candidates_size++] = i;
//             visited[i] = true;
//             // compute grad
//             proxl1PRrand::warm_start(i, stepSize, alpha, y, ds, dsinv, grad, ai, aj, visited, candidates, candidates_size, ra, c);
//         }
//     }
    
//     timeStamp1 = clock();

//     for (vtype i = 0; i < num_nodes; i ++) indices[i] = 0;
    
//     timeStamp2 = clock();
    
//     cout << "time initialization indices: " << (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC << endl;
    
//     timeStamp1 = clock();
    
//     for(auto it = nz_ids.begin() ; it != nz_ids.end(); ++it){
//         vtype i = *it;
//         grad[i] += q[i]/stepSize;
//     }
    
    for(itype j = ai[seed[0]]; j < ai[seed[0]+1]; j ++) grad[aj[j]] = 0;
    
//     timeStamp2 = clock();
    
//     cout << "time initialization gradient: " << (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC << endl;
    
//     for (vtype i = 0; i < num_nodes; ++i) {
//         cout << "1st grad[" << i << "]: " << grad[i] << endl;
//     }
    
    // exp start write graph
    // proxl1PRrand::writeGraph(num_nodes, "/home/c55hu/Documents/research/experiment/output/graph.txt", ai, aj);
    // proxl1PRrand::writeQdiag(num_nodes, "/home/c55hu/Documents/research/experiment/output/Qdiag.txt", ai, aj, d, alpha);
    // exp end
    double threshold = (1+epsilon)*rho*alpha;
    vtype numiter = 1;
    // some constant
    // maxiter *= 100;
    
//     timeStamp1 = clock();
    
//     vtype idx;
    
    while (Q.size()>0 && numiter<maxiter) {
        
//         for (pair<vtype, double> r : values) {
        node = Q.front();
        Q.pop();
            
//         for (vtype i = 0; i < num_nodes; ++i) {
//             cout  << "iter.: " << numiter << ", before q[" << i << "]: " << q[i] << endl;
//         }

//         for (vtype i = 0; i < num_nodes; ++i) {
//             cout  << "iter.: " << numiter << ", before grad[" << i << "]: " << grad[i] << endl;
//         }

//         timeStamp1 = clock();
//         r = dist(e2) % candidates_size;
//         timeStamp2 = clock();

//         sum_random = sum_random + (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC;

//        cout<< "Quicksort time "<< (float)(clock2 - clock1)/ CLOCKS_PER_SEC << " "<<endl;;


//             timeStamp1 = clock();
//         proxl1PRrand::updateGrad(r, stepSize, c, ra, &values, &grad, ds, dsinv, ai, aj, a, stepszra, &Q, threshold);
        
        
        dqs = -grad.at(node)-ds[node]*ra;
        dq = dqs*stepSize;
        cdqdsinv = c*dq*dsinv[node];   
        values.at(node) += dq;
        grad.at(node) += dqs;
        
        for (itype j = ai[node]; j < ai[node + 1]; ++j) {
            neighbor = aj[j];
            
            temp = -threshold*ds[neighbor];
            temp2 = cdqdsinv*dsinv[neighbor];
            
            got = grad.find(neighbor);
            
            if (got == grad.end()) {
//                 (*grad)[neighbor] = cdqdsinv*dsinv[neighbor]*a[j]; 
                grad[neighbor] = -temp2; 
                if (grad.at(neighbor) < temp) {
                    Q.push(neighbor);
                    values[neighbor] = 0;
                }
            }
            else {
//                 grad->at(neighbor) -= cdqdsinv*dsinv[neighbor]*a[j];
                grad_old = got->second;
                grad.at(neighbor) -= temp2;
                if ((grad.at(neighbor) < temp) && (grad_old >= temp)) {
                    Q.push(neighbor);
                    if (values.find(neighbor) == values.end()) values[neighbor] = 0;
                }
            }
        }
        
        
//         sum_ = 0;

//         for (pair<vtype, double> element : grad)
//         {
//             sum_ += -ds[element.first]*element.second/alpha;
//         }

//         for (pair<vtype, double> element : values)
//         {
//             sum_ += ds[element.first]*element.second;
//         }

//         cout << "sum: " << sum_ << endl;
        
//             timeStamp2 = clock();

//             sum_grad = sum_grad + (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC;

//         for (vtype i = 0; i < num_nodes; ++i) {
//             cout  << "iter.: " << numiter << ", after q[" << i << "]: " << q[i] << endl;
//         }

//         for (vtype i = 0; i < num_nodes; ++i) {
//             cout  << "iter.: " << numiter << ", after grad[" << i << "]: " << grad[i] << endl;
//         }

//             timeStamp1 = clock();
//         if (numiter % values.size() == 0) {
//             maxNorm = 0;
//             for (pair<vtype, double> it : values) {
//                 idx = it.first;
//                 maxNorm = max(maxNorm, abs(grad[idx]*dsinv[idx])); 
//             }
//         }
//             timeStamp2 = clock();

//             sum_term = sum_term + (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC;

        numiter++;

//             if (numiter++ > maxiter) {
//                 not_converged = 1;
//                 break;
//             }
//         }
    }
    
//     timeStamp2 = clock();
    
//     cout << "time loop: " << (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC << endl;
    
    //proxl1PRrand::writeTime(timeStamp, "/home/c55hu/Documents/research/experiment/output/time-rand.txt");
    //proxl1PRrand::writeLog(num_nodes, "/home/c55hu/Documents/research/experiment/output/q-rand.txt", q);
    // update y and q
    
//     cout << "sum_grad.: " << sum_grad << endl;
//     cout << "sum_term.: " << sum_term << endl;
//     cout << "sum_random.: " << sum_random << endl;
    
//     timeStamp1 = clock();
    
//     for (vtype i = 0; i < candidates_size; ++i) {
//         r = candidates[i];
//         q[r] *= ds[r];
//     }
    
    vtype counter = 0;
//     sum_ = 0;
    for (pair<vtype, double> it : values) {
        node = it.first;
        xids[counter] = node;
        q[counter] = values[node]*ds[node];
        counter++;
        
//         sum_ += values[node]*ds[node];
    }
    
//     cout << "sum: " << sum_ << endl;
    
//     sum_ = 0;

//     for (pair<vtype, double> element : grad)
//     {
//         sum_ += -ds[element.first]*element.second/alpha;
//     }

//     for (pair<vtype, double> element : values)
//     {
//         sum_ += ds[element.first]*element.second;
//     }

//     cout << "sum: " << sum_ << endl;
        
//     for (vtype i = 0; i < num_nodes; ++i) {
//         cout << "q[" << i << "]: " << q[i] << endl;
//     }
    
//     for (vtype i = 0; i < num_nodes; ++i) {
//         cout << "y[" << i << "]: " << y[i] << endl;
//     }
    
//     for (vtype i = 0; i < num_nodes; ++i) {
//         cout << "last grad[" << i << "]: " << grad[i] << endl;
//     }
    
//     timeStamp2 = clock();
    
//     cout << "time finalizing: " << (float)(timeStamp2 - timeStamp1)/ CLOCKS_PER_SEC << endl;
    
//     delete [] candidates;
//     delete [] visited;
    return values.size();
}

// template<typename vtype, typename itype>
// vtype graph<vtype,itype>::proxl1PRrand_unnormalized(vtype num_nodes, vtype* seed, vtype num_seeds, double epsilon, double alpha, double rho, double* q, double* d, double* ds, double* dsinv, double* grad, vtype maxiter)
// {
//     clock_t timeStamp = clock();
    
//     random_device rd;
//     mt19937_64 e2(rd());
//     uniform_int_distribution<long long int> dist(std::llround(std::pow(2,0)), std::llround(std::pow(2,63)));
    
// 	vtype not_converged = 0;
//     vtype* candidates = new vtype[num_nodes];
//     bool* visited = new bool[num_nodes];
//     for (vtype i = 0; i < num_nodes; ++i) visited[i] = false;
    
    
//     // initialize seed nodes as candidates
//     double maxNorm = 0;
//     vtype candidates_size = num_seeds;
//     for (vtype i = 0; i < num_seeds; ++i) {
//         // set gradient and update max norm
//         grad[seed[i]] = -alpha/num_seeds;
//         maxNorm = max(maxNorm, abs(grad[seed[i]]*d[seed[i]]));
//         // set as candidate nodes
//         candidates[i] = seed[i];
//         visited[seed[i]] = true;
//     }
    
//     double c = (1-alpha)/2;
//     double ra = rho*alpha;
//     double stepSize_const = 2.0/(1+alpha);

//     for(vtype i = 0; i < num_seeds; i ++){
//         grad[seed[i]] = -alpha/num_seeds;
//     }

//     //Find nonzero indices in y and dsinv
//     unordered_map<vtype,vtype> indices;
//     unordered_set<vtype> nz_ids;
    
//     for (vtype i = 0; i < num_nodes; i ++) {
//         if (q[i] != 0 && dsinv[i] != 0) {
//             indices[i] = 0;
//         }
//         if (q[i] != 0 || grad[i] != 0) {
//             nz_ids.insert(i);
//         }
//     }
    
    
//     for(auto it = nz_ids.begin() ; it != nz_ids.end(); ++it){
//         vtype i = *it;
//         grad[i] += q[i]*d[i]/stepSize_const;
//     }
//     vtype temp;

//     for(auto it = indices.begin() ; it != indices.end(); ++it){
//         vtype i = it->first;
//         for(itype j = ai[i]; j < ai[i+1]; j ++){
//             temp = aj[j];
//             grad[temp] -= q[i] * a[j] * c;
            
//             if (!visited[temp] && q[temp] - stepSize_const*(grad[temp]/d[temp]) >= stepSize_const*ra) {
//                 visited[temp] = true;
//                 candidates[candidates_size++] = temp;
//             }
//         }
//     }
// //     double maxNorm;
// //     grad[Seed] = -alpha;  // grad = -gradient
// //     maxNorm = abs(grad[Seed]/d[Seed]);
// //     vtype* candidates = new vtype[num_nodes];
// //     bool* visited = new bool[num_nodes];
// //     for (vtype i = 0; i < num_nodes; ++i) visited[i] = false;
// //     vtype candidates_size = 1;
// //     candidates[0] = Seed;
// //     visited[Seed] = true;
    
    
//     // exp start write graph
//     // proxl1PRrand::writeGraph(num_nodes, "/home/c55hu/Documents/research/experiment/output/graph.txt", ai, aj);
//     // proxl1PRrand::writeQdiag(num_nodes, "/home/c55hu/Documents/research/experiment/output/Qdiag.txt", ai, aj, d, alpha);
//     // exp end
//     double threshold = (1+epsilon)*rho*alpha;
//     vtype numiter = 1;
//     // some constant
//     // maxiter *= 100;
//     //for (vtype i = 0; i < num_nodes; ++i) ds[i] *= ra;
//     while (maxNorm > threshold) {
        
//         vtype r = dist(e2) % candidates_size;
//         proxl1PRrand::updateGrad_unnormalized(candidates[r], stepSize_const, c, ra, q, grad, d, ai, aj, a, visited, candidates, candidates_size);
        
//         if (numiter % 1000 == 0) {
//             maxNorm = 0;
//             for (vtype i = 0; i < candidates_size; ++i) {
//                 r = candidates[i];
//                 maxNorm = max(maxNorm, abs(grad[r]*dsinv[r]));
// //             cout << "iter.: " << numiter << " maxNorm: " <<  maxNorm << endl;
//             }
//         }
        
//         if (numiter++ > maxiter) {
//             //cout << "not converged" << endl;
//             not_converged = 1;
//             break;
//         }
        
//         // double crit = proxl1PRrand::compute_l2_norm<vtype>(grad,n);
//         // cout << "iter.: " << numiter << " l2norm: " <<  crit << endl;        
        
//     }
//     //proxl1PRrand::writeTime(timeStamp, "/home/c55hu/Documents/research/experiment/output/time-rand.txt");
//     //proxl1PRrand::writeLog(num_nodes, "/home/c55hu/Documents/research/experiment/output/q-rand.txt", q);

// //     for (vtype i = 0; i < num_nodes; ++i) y[i] = q[i];
    
//     delete [] candidates;
//     delete [] visited;
//     return not_converged;
// }

uint32_t proxl1PRrand32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double alpha,
                         double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                         double* dsinv, double epsilon, double* p,
                         uint32_t maxiter, uint32_t offset, double max_time,bool normalized_objective,
                         uint32_t* candidates)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,NULL);
    uint32_t actual_length = g.proxl1PRrand(n, v, v_nums, epsilon, alpha, rho, p, d, ds, dsinv, maxiter, candidates);
    return actual_length;
}

int64_t proxl1PRrand64(int64_t n, int64_t* ai, int64_t* aj, double* a, double alpha,
                        double rho, int64_t* v, int64_t v_nums, double* d, double* ds,
                        double* dsinv,double epsilon, double* p,
                        int64_t maxiter, int64_t offset, double max_time,bool normalized_objective,
                        int64_t* candidates)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    int64_t actual_length = g.proxl1PRrand(n, v, v_nums, epsilon, alpha, rho, p, d, ds, dsinv, maxiter, candidates);
    return actual_length;
}

uint32_t proxl1PRrand32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, double alpha,
                            double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                            double* dsinv, double epsilon, double* p,
                            uint32_t maxiter, uint32_t offset, double max_time,bool normalized_objective,
                            uint32_t* candidates)
{    
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    uint32_t actual_length = g.proxl1PRrand(n, v, v_nums, epsilon, alpha, rho, p, d, ds, dsinv, maxiter, candidates);
    return actual_length;
}
