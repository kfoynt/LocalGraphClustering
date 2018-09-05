/* Compute cut size, number of edges given a set and a graph */

#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <algorithm>
#include <stdint.h>

#include "include/setscores_c_interface.h"
#include "include/routines.hpp"

using namespace std;

void set_scores32(
		uint32_t n, uint32_t* ai, uint32_t* aj, uint32_t offset,
		uint32_t* R, uint32_t nR, uint32_t* voltrue, uint32_t* cut)
{
	graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    unordered_map<uint32_t, uint32_t> R_map;
    for (size_t i = 0; i < nR; i ++) {
    	R_map[R[i]];
    }
	pair<uint32_t, uint32_t> set_stats = g.get_stats(R_map, nR);
	voltrue[0] = get<0>(set_stats);
	cut[0] = get<1>(set_stats);
}

void set_scores32_64(
		uint32_t n, int64_t* ai, uint32_t* aj, uint32_t offset,
		uint32_t* R, uint32_t nR, int64_t* voltrue, int64_t* cut)
{
	graph<uint32_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    unordered_map<uint32_t, uint32_t> R_map;
    for (size_t i = 0; i < nR; i ++) {
    	R_map[R[i]];
    }
	pair<int64_t, int64_t> set_stats = g.get_stats(R_map, nR);
	voltrue[0] = get<0>(set_stats);
	cut[0] = get<1>(set_stats);
}

void set_scores64(
		int64_t n, int64_t* ai, int64_t* aj, int64_t offset,
		int64_t* R, int64_t nR, int64_t* voltrue, int64_t* cut)
{
	graph<int64_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    unordered_map<int64_t, int64_t> R_map;
    for (size_t i = 0; i < nR; i ++) {
    	R_map[R[i]];
    }
	pair<int64_t, int64_t> set_stats = g.get_stats(R_map, nR);
	voltrue[0] = get<0>(set_stats);
	cut[0] = get<1>(set_stats);
}

void set_scores_weighted32(
		uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double* degrees, uint32_t offset,
		uint32_t* R, uint32_t nR, double* voltrue, double* cut)
{
	graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,degrees);
    unordered_map<uint32_t, uint32_t> R_map;
    for (size_t i = 0; i < nR; i ++) {
    	R_map[R[i]];
    }
	pair<double, double> set_stats = g.get_stats_weighted(R_map, nR);
	voltrue[0] = get<0>(set_stats);
	cut[0] = get<1>(set_stats);
}

void set_scores_weighted32_64(
		uint32_t n, int64_t* ai, uint32_t* aj, double* a, double* degrees, uint32_t offset,
		uint32_t* R, uint32_t nR, double* voltrue, double* cut)
{
	graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,degrees);
    unordered_map<uint32_t, uint32_t> R_map;
    for (size_t i = 0; i < nR; i ++) {
    	R_map[R[i]];
    }
	pair<double, double> set_stats = g.get_stats_weighted(R_map, nR);
	voltrue[0] = get<0>(set_stats);
	cut[0] = get<1>(set_stats);
}

void set_scores_weighted64(
		int64_t n, int64_t* ai, int64_t* aj, double* a, double* degrees, int64_t offset,
		int64_t* R, int64_t nR, double* voltrue, double* cut)
{
	graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,degrees);
    unordered_map<int64_t, int64_t> R_map;
    for (size_t i = 0; i < nR; i ++) {
    	R_map[R[i]];
    }
	pair<double, double> set_stats = g.get_stats_weighted(R_map, nR);
	voltrue[0] = get<0>(set_stats);
	cut[0] = get<1>(set_stats);
}


