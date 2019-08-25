#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <stdint.h>
#include <typeinfo>
#include <sstream>
#include <stdio.h>

#include "../include/SimpleLocal_c_interface.h"
#include "../include/readData.hpp"


using namespace std;

int main()
{
    cout << "test SimpleLocal on file senate.smat with 0 offset" << endl;

    //Read and convert data
    string filename;
    filename = "../../graph/senate.smat";
    //filename = "../../graph/BrainSubgraph.smat";
    uint32_t m = 0, n = 0;
    uint32_t* ai = NULL, *aj = NULL;
    double* a = NULL;
    read_and_convert<uint32_t, uint32_t>(filename.c_str(), &m, &n, &ai, &aj, &a);

    //Read seed
    filename = "../../graph/senate_R.smat";
    stringstream ss;
    uint32_t nR = 0;
    uint32_t* R = NULL;
    read_seed<uint32_t, uint32_t>(filename.c_str(), &nR, &R);
	
    uint32_t* ret_set = (uint32_t*)malloc(sizeof(uint32_t)*m);

    //Begin calling C function
    cout << "calling C function" << endl;
    uint32_t actual_length = SimpleLocal32(m, nR, ai, aj, 0, R, ret_set, 0.3, false);
    cout << "output set" << endl;
    for(int i = 0; i < actual_length; i ++){
        cout << ret_set[i] << " ";
    }
    cout << endl << "total number of vertices is " << actual_length << endl;
    free(R);
    free(ai);
    free(aj);
    free(a);

    /*
    //Check the output
    cout << "compare the output with correct results" << endl;
    filename = "correct_output/MQI/minnesota_results.smat";
    char* read_file = readSMAT(filename.c_str());
    ss << read_file;
    free(read_file);
    uint32_t correct_length;
    ss >> correct_length;
    uint32_t* correct_ret_set = (uint32_t *)malloc(sizeof(uint32_t) * correct_length);
    for(size_t i = 0; i < (size_t)correct_length; i ++){
        ss >> correct_ret_set[i];
    }
    ss.str("");
    if(actual_length != correct_length){
        cout << "output length is not correct!" << endl;
        return EXIT_FAILURE;
    }
    else{
        for(size_t i = 0; i < (size_t)correct_length; i ++){
            if(ret_set[i] != correct_ret_set[i]){
                cout << "output is not correct!" << endl;
                return EXIT_FAILURE;
            }
        }
    }
    cout << "output is correct!" << endl;
    free(correct_ret_set);
    free(ret_set);
     
     */
	return EXIT_SUCCESS;
}
