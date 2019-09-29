#ifndef PROXL1PRRAND_C_INTERFACE
#define PROXL1PRRAND_C_INTERFACE

#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif

    uint32_t proxl1PRrand32(uint32_t n, uint32_t* ai, uint32_t* aj, double* a, double alpha,
                         double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                         double* dsinv, double epsilon, double* p,
                         uint32_t maxiter, uint32_t offset, double max_time, bool normalized_objective,
                         uint32_t* candidates);

    int64_t proxl1PRrand64(int64_t n, int64_t* ai, int64_t* aj, double* a, double alpha,
                            double rho, int64_t* v, int64_t v_nums, double* d, double* ds,
                            double* dsinv,double epsilon, double* p,
                            int64_t maxiter, int64_t offset, double max_time, bool normalized_objective,
                            int64_t* candidates);

    uint32_t proxl1PRrand32_64(uint32_t n, int64_t* ai, uint32_t* aj, double* a, double alpha,
                                double rho, uint32_t* v, uint32_t v_nums, double* d, double* ds,
                                double* dsinv, double epsilon, double* p,
                                uint32_t maxiter, uint32_t offset, double max_time, bool normalized_objective,
                                uint32_t* candidates);

#ifdef __cplusplus
}
#endif

#endif  // PROXL1PRRAND_C_INTERFACE defined