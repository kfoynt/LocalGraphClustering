#ifndef graph_check_hpp
#define graph_check_hpp

#include <stdio.h>

template<typename vtype, typename itype>
bool check_symmetric(itype* ai, vtype* aj, itype m, vtype n);

template<typename vtype, typename itype>
bool check_repeated(itype* ai, vtype* aj, itype m, vtype n);

template<typename vtype, typename itype>
bool check_diagonal(itype* ai, vtype* aj, itype m, vtype n);

#endif
