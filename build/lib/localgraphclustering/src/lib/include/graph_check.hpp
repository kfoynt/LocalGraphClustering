#ifndef GRAPH_CHECK_HPP
#define GRAPH_CHECK_HPP

#include <stdio.h>

template<typename vtype, typename itype>
bool check_symmetric(vtype* ei, vtype* ej, itype m, vtype n);

template<typename vtype, typename itype>
bool check_repeated(vtype* ei, vtype* ej, itype m, vtype n);

template<typename vtype, typename itype>
bool check_diagonal(vtype* ei, vtype* ej, itype m);

#include "../graph_check.cpp"

#endif
