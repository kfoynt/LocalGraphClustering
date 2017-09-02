#ifndef READ_DATA_HPP
#define READ_DATA_HPP

#include <iostream>

using namespace std;

template<typename vtype, typename itype>
void readList(const char* filename, vtype* m, itype* n, vtype** ei, vtype** ej, double** w);

template<typename vtype, typename itype>
void list_to_CSR(vtype m, itype n, vtype* ei, vtype* ej, double* w, 
        itype* ai, vtype* aj, double* a);

char* readSMAT(const char* filename);

template<typename vtype, typename itype>
void read_and_convert(const char* filename, vtype* nverts, itype* nedges,
                      itype** ret_ai, vtype** ret_aj, double** ret_a);

template<typename vtype, typename itype>
void read_seed(const char* filename, vtype* n, vtype** ids);

#include "../graph_lib_test/readData.cpp"
#endif
