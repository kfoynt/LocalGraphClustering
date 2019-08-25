#include "include/SimpleLocal_c_interface.h"
#include "include/SimpleLocal.h"

int64_t SimpleLocal64(int64_t n, int64_t nR, int64_t* ai, int64_t* aj, int64_t offset, int64_t* R, int64_t* ret_set, double delta, bool relcondflag)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.SimpleLocal(nR,R, ret_set, delta, relcondflag);
}

uint32_t SimpleLocal32(uint32_t n, uint32_t nR, uint32_t* ai, uint32_t* aj, uint32_t offset, uint32_t* R, uint32_t* ret_set, double delta, bool relcondflag)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.SimpleLocal(nR,R, ret_set, delta, relcondflag);
}

uint32_t SimpleLocal32_64(uint32_t n, uint32_t nR, int64_t* ai, uint32_t* aj, uint32_t offset, uint32_t* R, uint32_t* ret_set, double delta, bool relcondflag)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,NULL,offset,NULL);
    return g.SimpleLocal(nR,R, ret_set, delta, relcondflag);
}
