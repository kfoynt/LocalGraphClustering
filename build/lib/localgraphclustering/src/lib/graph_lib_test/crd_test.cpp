#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <stdint.h>
#include <typeinfo>
#include <sstream>
#include <stdio.h>
#include <time.h>

#include "../include/crd_c_interface.h"
#include "../include/readData.hpp"


using namespace std;

int main()
{
    clock_t t;
    cout << "test CRD on file JohnHopkins.smat with 0 offset" << endl;

    //Read and convert data
    string filename;
    filename = "../../graph/JohnHopkins.smat";
    int64_t m = 0, n = 0;
    int64_t* ai = NULL, *aj = NULL;
    double* a = NULL;
    read_and_convert<int64_t, int64_t>(filename.c_str(), &n, &m, &ai, &aj, &a);

    int64_t ref_node[] = {3215};
    //int64_t ref_node[] = {100};
    int64_t ref_node_size = 1;
    int64_t* cut = (int64_t*)malloc(sizeof(int64_t)*n); 
    int64_t offset = 0;
    int64_t U = 3;
    int64_t h = 10;
    int64_t w = 2;
    int64_t iterations = 20;

    //Begin calling C function
    cout << "calling C function" << endl;
    t = clock();
    int64_t actual_length = capacity_releasing_diffusion64(n, ai, aj, a, offset, cut, U, h, w, 
                            iterations, ref_node, ref_node_size);
    t = clock() - t;
    printf ("It took me %lu clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
    cout << "output set" << endl;
    for(int i = 0; i < actual_length; i ++){
        cout << cut[i] << " ";
    }
    cout << endl << "total number of vertices is " << actual_length << endl;
    free(a);
    free(ai);
    free(aj);
    free(cut);

    return EXIT_SUCCESS;
}