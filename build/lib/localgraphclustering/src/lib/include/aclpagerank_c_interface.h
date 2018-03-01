#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif
    uint32_t aclpagerank32(
        uint32_t n, uint32_t* ai, uint32_t* aj, uint32_t offset,
            //Compressed sparse row representation, with offset for
            //zero based (matlab) or one based arrays (julia)
        double alpha,    //value of alpha
        double eps,    //value of epsilon
        uint32_t* seedids, uint32_t nseedids,    //the set indices for seeds
        uint32_t maxsteps,    //the maximum number of steps
        uint32_t* xids, uint32_t xlength, double* values); //solution vectors


    int64_t aclpagerank64(
        int64_t n, int64_t* ai, int64_t* aj, int64_t offset,
            //Compressed sparse row representation, with offset for
            //zero based (matlab) or one based arrays (julia)
        double alpha,    //value of alpha
        double eps,    //value of epsilon
        int64_t* seedids, int64_t nseedids,    //the set indices for seeds
        int64_t maxsteps,    //the maximum number of steps
        int64_t* xids, int64_t xlength, double* values); //solution vectors

    uint32_t aclpagerank32_64(
        uint32_t n, int64_t* ai, uint32_t* aj, uint32_t offset, 
            //Compressed sparse row representation, with offset for
            //zero based (matlab) or one based arrays (julia)
        double alpha,   //value of alpha
        double eps,    //value of epsilon
        uint32_t* seedids, uint32_t nseedids,     //the set indices for seeds
        uint32_t maxsteps,   //the maximum number of steps
        uint32_t* xids, uint32_t xlength, double* values);  //solution vectors

#ifdef __cplusplus
}
#endif



