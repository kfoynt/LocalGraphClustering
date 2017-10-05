#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

double densest_subgraph32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, 
                            uint32_t offset, uint32_t* ret_set, uint32_t* actual_length);

double densest_subgraph32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, 
                            uint32_t offset, uint32_t* ret_set, uint32_t* actual_length);

double densest_subgraph64(int64_t n, int64_t* ai, int64_t* aj, double* a, 
                            int64_t offset, int64_t* ret_set, int64_t* actual_length);
#ifdef __cplusplus
}
#endif

