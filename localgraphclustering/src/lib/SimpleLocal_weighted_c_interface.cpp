#include "include/SimpleLocal_weighted_c_interface.h"
#include "include/SimpleLocal_weighted.h"

int64_t SimpleLocal_weighted64(int64_t n, int64_t nR, int64_t* ai, int64_t* aj, double* a, double* degrees, int64_t offset, int64_t* R, int64_t* ret_set, double delta, bool relcondflag)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.SimpleLocal_weighted(nR,R, ret_set, delta, relcondflag);
}

uint32_t SimpleLocal_weighted32(uint32_t n, uint32_t nR, uint32_t* ai, uint32_t* aj, double* a, double* degrees, uint32_t offset, uint32_t* R, uint32_t* ret_set, double delta, bool relcondflag)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.SimpleLocal_weighted(nR,R, ret_set, delta, relcondflag);
}

uint32_t SimpleLocal_weighted32_64(uint32_t n, uint32_t nR, int64_t* ai, uint32_t* aj, double* a, double* degrees, uint32_t offset, uint32_t* R, uint32_t* ret_set, double delta, bool relcondflag)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,degrees);
    return g.SimpleLocal_weighted(nR,R, ret_set, delta, relcondflag);
}
