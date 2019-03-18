#include "include/MQI_weighted_c_interface.h"
#include "include/MQI_weighted.h"


using namespace std;

int64_t MQI_weighted64(int64_t n, int64_t nR, int64_t* ai, int64_t* aj, double* a, double* degrees, int64_t offset, int64_t* R, int64_t* ret_set)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.MQI_weighted(nR,R,ret_set);
}

int32_t MQI_weighted32(int32_t n, int32_t nR, int32_t* ai, int32_t* aj, double* a, double* degrees, int32_t offset, int32_t* R, int32_t* ret_set)
{
    graph<int32_t,int32_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.MQI_weighted(nR,R, ret_set);
}

int32_t MQI_weighted32_64(int32_t n, int32_t nR, int64_t* ai, int32_t* aj, double* a, double* degrees, int32_t offset, int32_t* R, int32_t* ret_set)
{
    graph<int32_t,int64_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.MQI_weighted(nR,R, ret_set);
}
