#include "include/MQI_weighted_c_interface.h"
#include "include/MQI_weighted.h"
#include <iostream>


using namespace std;

int64_t MQI_weighted64(int64_t n, int64_t nR, int64_t* ai, int64_t* aj, double* a, double* degrees, int64_t offset, int64_t* R, int64_t* ret_set)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.MQI_weighted(nR,R,ret_set);
}

uint32_t MQI_weighted32(uint32_t n, uint32_t nR, uint32_t* ai, uint32_t* aj, double* a, double* degrees, uint32_t offset, uint32_t* R, uint32_t* ret_set)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.MQI_weighted(nR,R, ret_set);
}

uint32_t MQI_weighted32_64(uint32_t n, uint32_t nR, int64_t* ai, uint32_t* aj, double* a, double* degrees, uint32_t offset, uint32_t* R, uint32_t* ret_set)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.MQI_weighted(nR,R, ret_set);
}
