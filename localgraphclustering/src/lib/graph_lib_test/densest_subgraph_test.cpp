#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <inttypes.h>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include "../include/graph_check.hpp"
#include "../include/readData.hpp"
#include "../include/routines.hpp"
#include "../include/densest_subgraph_c_interface.h"

using namespace std;

int main()
{
    FILE *rptr = fopen("../../graph/Erdos02-cc.smat", "r");
    if(rptr == NULL) {
        fprintf(stderr, 
                "The file %s couldn't be opened for reading (Does it exist?)!\n", 
                "Erdos02-cc.smat");
        return EXIT_FAILURE;
    }

    /*read the content from the input file to a stringstream*/
    fseek(rptr, 0, SEEK_END);
    size_t fsize = ftell(rptr);
    char *read_file = (char *)malloc(sizeof(char) * fsize);
    if(read_file == NULL) {
        fprintf(stderr, "malloc failing!\n");
        fclose(rptr);
        return EXIT_FAILURE;
    }
    fseek(rptr, 0, SEEK_SET);
    size_t nread = fread(read_file, sizeof(char), fsize, rptr);
    if (nread != fsize) {
        fprintf(stderr, "read failed on %s\n", "Erdos02-cc.smat");
        return EXIT_FAILURE;
    }
    stringstream ss;
    ss << read_file;
    free(read_file);

    int64_t n, m;
    int64_t *ei, *ej;
    double *w;

    ss >> n;
    ss >> n;
    ss >> m;
    ei = (int64_t *)malloc(sizeof(int64_t) * m);
    ej = (int64_t *)malloc(sizeof(int64_t) * m);
    w = (double *)malloc(sizeof(double) * m);
    int64_t i;
    for(i = 0; i < m; i ++) {
        ss >> ei[i];
        ss >> ej[i];
        ss >> w[i];
        // TODO validate input
        if(ei[i] < 0 || ei[i] >= n || ej[i] < 0 || ej[i] >= n)
        {
            fprintf(stderr, "Invalid Input in Line %lld!\n", i + 2);
            fclose(rptr);
            free(ei);
            free(ej);
            free(w);
            return EXIT_FAILURE;
        }
        // TODO check for negative weights
        if(w[i] < 0)
        {
            fprintf(stderr, "Negative Weight in Line %lld!\n", i + 2);
            fclose(rptr);
            free(ei);
            free(ej);
            free(w);
            return EXIT_FAILURE;
        }
    }
    fclose(rptr);

    // TODO check for diagonal edges
    if(check_diagonal<int64_t,int64_t>(ei, ej, m))
    {
        free(ei);
        free(ej);
        free(w);
        return EXIT_FAILURE;
    }

    // TODO check for repeated edges
    if(check_repeated<int64_t,int64_t>(ei, ej, m, n))
    {
        free(ei);
        free(ej);
        free(w);
        return EXIT_FAILURE;
    }
    // TODO check symmetric
    if(check_symmetric<int64_t,int64_t>(ei, ej, m, n))
    {
        free(ei);
        free(ej);
        free(w);
        return EXIT_FAILURE;
    }
    int64_t* ai = new int64_t[n+1];
    int64_t* aj = new int64_t[m];
    double* a = new double[m];

    list_to_CSR(n, m, ei, ej, w, ai, aj, a);

    fprintf(stderr, "loaded file %s : %12" PRId64 " nodes   %12" PRId64 " edges\n",
            "Erdos02-cc.smat", n, m);

    double ret;
    int64_t *ret_set = (int64_t *)malloc(sizeof(int64_t) * n);
    int64_t actual_length = n;
    int64_t offset = 0;
    ret = densest_subgraph64(n, ai, aj, a, offset, ret_set, &actual_length);
    cout << ret << endl;
    cout << actual_length << endl;
    for (size_t j = 0; j < actual_length; ++j) {
        cout << ret_set[j] << endl;
    }

    delete[] ai;
    delete[] aj;
    delete[] a;
    free(ret_set);

    return 0;
}
