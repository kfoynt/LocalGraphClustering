#ifdef GRAPH_CHECK_HPP

#include <unordered_set>
#include <iostream>
#include <vector>
#include "include/graph_check.hpp"

using namespace std;

#define TRUE 0
#define FALSE 1

template<typename vtype, typename itype>
bool check_symmetric(vtype *ei, vtype *ej, itype m, vtype n)
{
    vector< unordered_set<vtype> > graph(n);
    
    for (size_t i = 0; i < (size_t)m; i ++) {
        graph[ei[i]].insert(ej[i]);
    }
    
    for (size_t i = 0; i < (size_t)m; i ++) {
        if (graph[ej[i]].count(ei[i]) == 0) {
            fprintf(stderr, "Symmetric Error in Line %lld\n!", i + 2);
            return FALSE;
        }
    }
    
    return TRUE;
}

template<typename vtype, typename itype>
bool check_repeated(vtype *ei, vtype *ej, itype m, vtype n)
{
    vector< unordered_set<vtype> > graph(n);
    
    for (size_t i = 0; i < (size_t)m; i ++) {
        if (graph[ei[i]].count(ej[i]) != 0) {
            fprintf(stderr, "Repeated Edges in Line %lld\n!", i + 2);
            return FALSE;
        }
        graph[ei[i]].insert(ej[i]);
    }
    
    return TRUE;
}

template<typename vtype, typename itype>
bool check_diagonal(vtype *ei, vtype *ej, itype m)
{
    for(size_t i = 0; i < (size_t)m; i ++)
    {
        if(ei[i] == ej[i])
        {
            fprintf(stderr, "Diagonal Edge in Line %lld\n!", i + 2);
            return FALSE;
        }
    }
    return TRUE;
}

#endif

