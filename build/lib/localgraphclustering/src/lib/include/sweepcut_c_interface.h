#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif
    uint32_t sweepcut_without_sorting32(uint32_t* ids, uint32_t* results, uint32_t num, 
            uint32_t n, uint32_t* ai, uint32_t* aj, double* a, uint32_t offset, double* ret_cond, double* degrees);
    int64_t sweepcut_without_sorting64(int64_t* ids, int64_t* results, int64_t num, 
            int64_t n, int64_t* ai, int64_t* aj, double* a, int64_t offset, double* ret_cond, double* degrees); 
    uint32_t sweepcut_without_sorting32_64(uint32_t* ids, uint32_t* results, uint32_t num, 
            uint32_t n, int64_t* ai, uint32_t* aj, double* a, uint32_t offset, double* ret_cond, double* degrees); 

    uint32_t sweepcut_with_sorting32(double* value, uint32_t* ids, uint32_t* results, 
            uint32_t num, uint32_t n, uint32_t* ai, uint32_t* aj, double* a, uint32_t offset, double* ret_cond, double* degrees); 
    int64_t sweepcut_with_sorting64(double* value, int64_t* ids, int64_t* results, 
            int64_t num, int64_t n, int64_t* ai, int64_t* aj, double* a, int64_t offset, double* ret_cond, double* degrees); 
    uint32_t sweepcut_with_sorting32_64(double* value, uint32_t* ids, uint32_t* results, 
            uint32_t num, uint32_t n, int64_t* ai, uint32_t* aj, double* a, uint32_t offset, double* ret_cond, double* degrees); 

#ifdef __cplusplus
}
#endif



