#include "include/SimpleLocal_c_interface.h"
#include "include/SimpleLocal.h"

int64_t SimpleLocal64(int64_t n, int64_t nR, int64_t* ai, int64_t* aj, int64_t offset, int64_t* R, int64_t* ret_set, double delta)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.SimpleLocal(nR,R, ret_set, delta);
}

int32_t SimpleLocal32(int32_t n, int32_t nR, int32_t* ai, int32_t* aj, int32_t offset, int32_t* R, int32_t* ret_set, double delta)
{
    graph<int32_t,int32_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.SimpleLocal(nR,R, ret_set, delta);
}

int32_t SimpleLocal32_64(int32_t n, int32_t nR, int64_t* ai, int32_t* aj, int32_t offset, int32_t* R, int32_t* ret_set, double delta)
{
    graph<int32_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.SimpleLocal(nR,R, ret_set, delta);
}
