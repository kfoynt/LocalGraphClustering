#ifndef _PPR_PATH_C_INTERFACE_H_
#define _PPR_PATH_C_INTERFACE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    struct path_info{
        int64_t* num_eps;
        double* epsilon;
        double* conds;
        double* cuts;
        double* vols;
        int64_t* setsizes;
        int64_t* stepnums;
    };

    struct rank_info{
        int64_t* starts;
        int64_t* ends;
        int64_t* nodes;
        int64_t* deg_of_pushed;
        int64_t* size_of_solvec;
        int64_t* size_of_r;
        double* val_of_push;
        double* global_bcond;

        int64_t* nrank_changes;
        int64_t* nrank_inserts;
        int64_t* nsteps;
        int64_t* size_for_best_cond;
    };
    
    int64_t ppr_path64(int64_t n, int64_t* ai, int64_t* aj, int64_t offset, double alpha, 
        double eps, double rho, int64_t* seedids, int64_t nseedids, int64_t* xids, 
        int64_t xlength, struct path_info ret_path_results, struct rank_info ret_rank_results);
    uint32_t ppr_path32(uint32_t n, uint32_t* ai, uint32_t* aj, uint32_t offset, double alpha, 
        double eps, double rho, uint32_t* seedids, uint32_t nseedids, uint32_t* xids, 
        uint32_t xlength, struct path_info ret_path_results, struct rank_info ret_rank_results);
    uint32_t ppr_path32_64(uint32_t n, int64_t* ai, uint32_t* aj, uint32_t offset, double alpha, 
        double eps, double rho, uint32_t* seedids, uint32_t nseedids, uint32_t* xids, 
        uint32_t xlength, struct path_info ret_path_results, struct rank_info ret_rank_results);

#ifdef __cplusplus
}
#endif

#endif
