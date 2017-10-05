#include "include/densest_subgraph_c_interface.h"
#include "include/densest_subgraph.h"

double densest_subgraph32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, 
                            uint32_t offset, uint32_t* ret_set, uint32_t* actual_length)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,NULL);
    double density = g.densest_subgraph(ret_set,actual_length);
    return density;
}

double densest_subgraph32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, 
                            uint32_t offset, uint32_t* ret_set, uint32_t* actual_length)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    double density = g.densest_subgraph(ret_set,actual_length);
    return density;
}

double densest_subgraph64(int64_t n, int64_t* ai, int64_t* aj, double* a, 
                            int64_t offset, int64_t* ret_set, int64_t* actual_length)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    double density = g.densest_subgraph(ret_set,actual_length);
    return density;
}

