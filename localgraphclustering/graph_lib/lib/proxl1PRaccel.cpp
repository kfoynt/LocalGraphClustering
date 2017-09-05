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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <stdint.h>
#include <cmath>
#include <vector>
#include <time.h>

#include "include/proxl1PRaccel_c_interface.h"
#include "include/routines.hpp"

using namespace std;

    template<typename vtype>
double find_max(double* grad, double* ds, vtype n){
    double max_num = 0;
    for(vtype i = 0; i < n; i ++){
        //cout << max_num << " " << grad[i]/ds[i] << endl;
        max_num = max(max_num,abs(grad[i]/ds[i]));
    }
    return max_num;
}

    template<typename vtype, typename itype>
void update_grad(double* grad, vector<double> y, vector<double> c, itype* ai, vtype* aj, double* a,
                 vtype n, double alpha, double* dsinv, vtype offset)
{
    for(vtype i = 0; i < n; i ++){
        grad[i] = (1+alpha)/2*y[i] - c[i];
    }
    for(vtype i = 0; i < n; i ++){
        for(itype j = ai[i]-offset; j < ai[i+1]-offset; j ++){
            grad[i] -= a[j] * y[aj[j]-offset] * dsinv[i] * dsinv[aj[j]-offset] * (1-alpha)/2 * 0.5;
            grad[aj[j]-offset] -= a[j]* y[i] * dsinv[i] * dsinv[aj[j]-offset] * (1-alpha)/2 * 0.5;
        }
        
    }
}

    template<typename vtype, typename itype>
vtype graph<vtype,itype>::proxl1PRaccel(double alpha, double rho, vtype* v, vtype v_nums, double* d,
                                        double* ds, double* dsinv, double epsilon, double* grad, double* p,
                                        vtype maxiter,double max_time)
{
    /*cout << "dsinv" << endl;
    for(vtype i = 0; i < n; i ++){
        cout << dsinv[i] << endl;
    }
    
    cout << "a" << endl;
    for(vtype i = 0; i < 8; i ++){
        cout << a[i] << endl;
    }*/
    clock_t t1,t2;
    vtype not_converged = 0;
    vector<double> c (n,0);
    for(vtype i = 0; i < v_nums; i ++){
        grad[v[i]-offset] = -1 * alpha / (v_nums * ds[v[i]-offset]);
        c[v[i]-offset] = -1 * grad[v[i]-offset];
    }
    /*for(vtype i = 0; i < n; i ++){
        cout << grad[i] << endl;
    }*/
    vector<double> q (n,0);
    vector<double> q_old (n,0);
    vector<double> y (n,0);
    double z;
    size_t iter = 0;
    double thd = (1 + epsilon) * rho * alpha;
    //cout << thd << " " << find_max<vtype>(grad,ds,n) << endl;
    double thd1,betak;
    t1 = clock();
    while((iter < (size_t)maxiter) && (find_max<vtype>(grad,ds,n) > thd)){
        iter ++;
        q_old = q;
        for(vtype i = 0; i < n; i ++){
            z = y[i] - grad[i];
            thd1 = rho*alpha*ds[i];
            if(z >= thd1){
                q[i] = z - thd1;
            }
            else if(z <= -1 * thd1){
                q[i] = z + thd1;
            }
            else{
                q[i] = 0;
            }
        }
        if(iter == 1){
            betak = 0;
        }
        else{
            betak = (1-sqrt(alpha))/(1+sqrt(alpha));
        }
        for(vtype i = 0; i < n; i ++){
            y[i] = q[i] + betak*(q[i]-q_old[i]);
        }
        update_grad(grad,y,c,ai,aj,a,n,alpha,dsinv,offset);
        
        /*if(iter == 1){
            cout << "y" << endl;
            for(vtype i = 0; i < n; i ++){
                cout << y[i] << endl;
            }
            cout << "q" << endl;
            for(vtype i = 0; i < n; i ++){
                cout << q[i] << endl;
            }
            cout << "c" << endl;
            for(vtype i = 0; i < n; i ++){
                cout << c[i] << endl;
            }
            cout << "grad" << endl;
            for(vtype i = 0; i < n; i ++){
                cout << grad[i] << endl;
            }
        }*/
        
        t2 = clock();
        if(((double)t2 - (double)t1)/double(CLOCKS_PER_SEC) > max_time){
            not_converged = 1;
            return not_converged;
        }
        /*if(iter == 1){
            cout << find_max<vtype>(grad,ds,n) << endl;
        }*/
    }
    
    if(iter >= (size_t)maxiter){
        not_converged = 1;
    }
    
    for(vtype i = 0; i < n; i ++){
        p[i] = abs(q[i])*ds[i];
    }
    return not_converged;
}

uint32_t proxl1PRaccel32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double alpha,
                         double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                         double* dsinv, double epsilon, double* grad, double* p, uint32_t maxiter, uint32_t offset,double max_time)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,NULL);
    return g.proxl1PRaccel(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,maxiter,max_time);
}

int64_t proxl1PRaccel64(int64_t n, int64_t* ai, int64_t* aj, double* a, double alpha,
                        double rho, int64_t* v, int64_t v_nums, double* d, double* ds,
                        double* dsinv,double epsilon, double* grad, double* p, int64_t maxiter, int64_t offset,double max_time)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    return g.proxl1PRaccel(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,maxiter,max_time);
}

uint32_t proxl1PRaccel32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, double alpha,
                            double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                            double* dsinv, double epsilon, double* grad, double* p,
                            uint32_t maxiter, uint32_t offset,double max_time)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    return g.proxl1PRaccel(alpha,rho,v,v_nums,d,ds,dsinv,epsilon,grad,p,maxiter,max_time);
}
