#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif
    void set_scores32(
        uint32_t n, uint32_t* ai, uint32_t* aj, uint32_t offset,
        uint32_t* R, uint32_t nR, uint32_t* voltrue, uint32_t* cut);
    void set_scores32_64(
        uint32_t n, int64_t* ai, uint32_t* aj, uint32_t offset,
        uint32_t* R, uint32_t nR, int64_t* voltrue, int64_t* cut);
    void set_scores64(
        int64_t n, int64_t* ai, int64_t* aj, int64_t offset,
        int64_t* R, int64_t nR, int64_t* voltrue, int64_t* cut);
    void set_scores_weighted32(
        uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double* degrees, uint32_t offset,
        uint32_t* R, uint32_t nR, double* voltrue, double* cut);
    void set_scores_weighted32_64(
        uint32_t n, int64_t* ai, uint32_t* aj, double* a, double* degrees, uint32_t offset,
        uint32_t* R, uint32_t nR, double* voltrue, double* cut);
    void set_scores_weighted64(
        int64_t n, int64_t* ai, int64_t* aj, double* a, double* degrees, int64_t offset,
        int64_t* R, int64_t nR, double* voltrue, double* cut);

#ifdef __cplusplus
}
#endif