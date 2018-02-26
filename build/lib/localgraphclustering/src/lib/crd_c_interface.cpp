#include "include/crd_c_interface.h"
#include "include/crd.h"

uint32_t capacity_releasing_diffusion32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, 
                            uint32_t offset, uint32_t* cut, uint32_t U, uint32_t h, uint32_t w, 
                            uint32_t iterations, uint32_t* ref_node, uint32_t ref_node_size)
{
    graph<uint32_t,uint32_t> g(ai[n],n,ai,aj,a,offset,NULL);
    vector<uint32_t> ref (ref_node,ref_node+ref_node_size);
    uint32_t cut_size = g.capacity_releasing_diffusion(ref,U,h,w,iterations,cut);
    return cut_size;
}

uint32_t capacity_releasing_diffusion32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, 
                            uint32_t offset, uint32_t* cut, uint32_t U, uint32_t h, uint32_t w, 
                            uint32_t iterations, uint32_t* ref_node, uint32_t ref_node_size)
{
    graph<uint32_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    vector<uint32_t> ref (ref_node,ref_node+ref_node_size);
    uint32_t cut_size = g.capacity_releasing_diffusion(ref,U,h,w,iterations,cut);
    return cut_size;
}

int64_t capacity_releasing_diffusion64(int64_t n, int64_t* ai, int64_t* aj, double* a, 
                            int64_t offset, int64_t* cut, int64_t U, int64_t h, int64_t w, 
                            int64_t iterations, int64_t* ref_node, int64_t ref_node_size)
{
    graph<int64_t,int64_t> g(ai[n],n,ai,aj,a,offset,NULL);
    vector<int64_t> ref (ref_node,ref_node+ref_node_size);
    int64_t cut_size = g.capacity_releasing_diffusion(ref,U,h,w,iterations,cut);
    return cut_size;
}