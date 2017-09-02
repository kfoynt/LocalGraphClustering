#include "graph_check.hpp"
#include <unordered_set>
#include <iostream>

template<typename vtype, typename itype>
bool check_symmetric(int64_t *ei, int64_t *ej, int64_t m, int64_t n)
{
    vector< unordered_set<int> > graph(n);
    
    for (int64_t i = 0; i < m; i ++) {
        graph[ei[i]].insert(ej[i]);
    }
    
    for (int64_t i = 0; i < m; i ++) {
        if (graph[ej[i]].count(ei[i]) == 0) {
            fprintf(stderr, "Symmetric Error in Line %lld\n!", i + 2);
            return FALSE;
        }
    }
    
    return TRUE;
}

bool check_repeated(int64_t *ei, int64_t *ej, int64_t m, int64_t n)
{
    vector< unordered_set<int> > graph(n);
    
    for (int64_t i = 0; i < m; i ++) {
        if (graph[ei[i]].count(ej[i]) != 0) {
            fprintf(stderr, "Repeated Edges in Line %lld\n!", i + 2);
            return FALSE;
        }
        graph[ei[i]].insert(ej[i]);
    }
    
    return TRUE;
}

bool check_diagonal(int64_t *ei, int64_t *ej, int64_t m)
{
    int64_t i;
    for(i = 0; i < m; i ++)
    {
        if(ei[i] == ej[i])
        {
            fprintf(stderr, "Diagonal Edge in Line %lld\n!", i + 2);
            return FALSE;
        }
    }
    return TRUE;
}



