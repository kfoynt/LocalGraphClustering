#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "../include/proxl1PRrand_c_interface.h"

using namespace std;

int main() {
    int64_t ai[5] = {0,3,6,7,11};
    int64_t aj[11] = {0,1,3,0,1,3,3,0,1,2,3};
    double a[11] = {1,1,1,1,1,1,1,1,1,1,1};
    double rho = 0.00001;
    double alpha = 0.15;
    int64_t v[2] = {1,2};
    double epsilon = 0.0001;
    int64_t maxiter = 10000;
    double* p = new double[4]();
    double* grad = new double[4]();
    int64_t n = 4;
    double* d = new double[n];
    double* ds = new double[n];
    double* dsinv = new double[n];
    for(int i = 0; i < n; i ++){
        d[i] = ai[i+1] - ai[i];
        ds[i] = sqrt(d[i]);
        dsinv[i] = 1/ds[i];
    }
    double p0[4] = {0,0,0,0};
    cout << proxl1PRrand64(n,ai,aj,a,alpha,rho,v,2,d,ds,dsinv,epsilon,grad,p,p0,maxiter,0,100, true) << endl;
    double sum = 0;
    cout << "p" << endl;
    for(int i = 0; i < 4; i ++){
        cout << p[i] << endl;
        sum += p[i];
    }
    cout << "PageRank vector sum: " << sum << endl;
    cout << "final grad" << endl;
    for(int i = 0; i < 4; i ++){
        cout << grad[i] << endl;
    }
    return EXIT_SUCCESS;
}