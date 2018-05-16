#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <algorithm>
#include <stdint.h>

#include "include/triangleclusters_c_interface.h"
#include "include/routines.hpp"

using namespace std;

template<typename vtype, typename itype>
void graph<vtype,itype>::triangleclusters(double* cond, double* cut, double* vol, 
	double* cc, double* t)
{
	double Gvol = 0;
	vector<bool> ind(n,false);
	for (vtype i = 0; i < n; i ++) {

		//index the neighbors
		for (itype j = ai[i]; j < ai[i+1]; j ++) {
			ind[aj[j]] = true;
		}
		ind[i] = true;

	    double d = (double)(ai[i+1]-ai[i]);
	    double curvol = 0;
	    double curcut = 0;
	    double curt = 0;
        
        //do two BFS steps
	    for (itype j = ai[i]; j < ai[i+1]; j ++) {
	    	vtype k = aj[j];
	    	if (i == k) {d -= 1.0; continue;}
	    	curvol += 1.0;
	    	Gvol += 1.0;
	    	for (itype j2 = ai[k]; j2 < ai[k+1]; j2++) {
	    		vtype x = aj[j2];
	    		if (x == k) {continue;}
	    		curvol += 1.0;
	    		if (x == i) {continue;}
	    		if (ind[x]) {
	    			curt += 1.0;
	    		} else {
	    			curcut += 1.0;
	    		}
	    	}
	    }
        
        // assign the output
	    cut[i] = curcut;
	    vol[i] = curvol;
	    if (d > 1) {
	    	cc[i] = curt/(d*(d-1));
	    } else {
	    	cc[i] = 0;
	    }
	    t[i] = curt/2;
        
        // clear the index
	    for (itype j = ai[i]; j < ai[i+1]; j ++) {
	    	ind[aj[j]] = false;
	    }
	    ind[i] = false;
	}

	for (vtype i = 0; i < n; i ++) {
		cond[i] = cut[i]/min(vol[i], Gvol-vol[i]);
	}
}

void triangleclusters32(
        uint32_t n, uint32_t* ai, uint32_t* aj,
        double* cond, double* cut, double* vol, 
	    double* cc, double* t, uint32_t offset)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    g.triangleclusters(cond, cut, vol, cc, t);
}

void triangleclusters64(
        int64_t n, int64_t* ai, int64_t* aj,
        double* cond, double* cut, double* vol, 
	    double* cc, double* t, int64_t offset)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    g.triangleclusters(cond, cut, vol, cc, t);
}

void triangleclusters32_64(
        uint32_t n, int64_t* ai, uint32_t* aj,
        double* cond, double* cut, double* vol, 
	    double* cc, double* t, uint32_t offset)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    g.triangleclusters(cond, cut, vol, cc, t);
}
