#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif
    void triangleclusters32(
        uint32_t n, uint32_t* ai, uint32_t* aj,
        double* cond, double* cut, double* vol, 
        double* cc, double* t, uint32_t offset);
    void triangleclusters64(
        int64_t n, int64_t* ai, int64_t* aj,
        double* cond, double* cut, double* vol, 
        double* cc, double* t, int64_t offset);
    void triangleclusters32_64(
        uint32_t n, int64_t* ai, uint32_t* aj,
        double* cond, double* cut, double* vol, 
        double* cc, double* t, uint32_t offset);
#ifdef __cplusplus
}
#endif