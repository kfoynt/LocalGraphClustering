#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    uint32_t capacity_releasing_diffusion32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, 
                            uint32_t offset, uint32_t* cut, uint32_t U, uint32_t h, uint32_t w, 
                            uint32_t iterations, uint32_t* ref_node, uint32_t ref_node_size);
    uint32_t capacity_releasing_diffusion32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, 
                            uint32_t offset, uint32_t* cut, uint32_t U, uint32_t h, uint32_t w, 
                            uint32_t iterations, uint32_t* ref_node, uint32_t ref_node_size);
    int64_t capacity_releasing_diffusion64(int64_t n, int64_t* ai, int64_t* aj, double* a, 
                            int64_t offset, int64_t* cut, int64_t U, int64_t h, int64_t w, 
                            int64_t iterations, int64_t* ref_node, int64_t ref_node_size);

#ifdef __cplusplus
}
#endif