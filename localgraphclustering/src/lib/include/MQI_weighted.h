#ifndef MQI_WEIGHTED_H
#define MQI_WEIGHTED_H

#include <utility>
#include <unordered_map>

using namespace std;

template<typename vtype, typename itype>
void build_map_weighted(itype* ai, vtype* aj, vtype offset, unordered_map<vtype, vtype>& R_map, 
        unordered_map<vtype, double>& degree_map, vtype& R, vtype nR, double* degrees);


#include "../MQI_weighted.cpp"
#endif
