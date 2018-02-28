#ifndef MQI_H
#define MQI_H

#include <utility>
#include <unordered_map>

using namespace std;

template<typename vtype, typename itype>
void build_map(itype* ai, vtype* aj, vtype offset, unordered_map<vtype, vtype>& R_map, 
        unordered_map<vtype, vtype>& degree_map, vtype& R, vtype nR);


#include "../MQI.cpp"
#endif
