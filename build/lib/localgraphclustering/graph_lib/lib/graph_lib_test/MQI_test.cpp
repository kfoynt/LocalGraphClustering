#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <stdint.h>
#include <typeinfo>
#include <sstream>
#include <stdio.h>

#include "../include/MQI_c_interface.h"
#include "../include/readData.hpp"


using namespace std;

int main()
{
    cout << "test MQI on file minnesota.smat with 0 offset" << endl;

    //Read and convert data
    string filename;
    filename = "../../graph/minnesota.smat";
    int64_t m = 0, n = 0;
    int64_t* ai = NULL, *aj = NULL;
    double* a = NULL;
    read_and_convert<int64_t, int64_t>(filename.c_str(), &m, &n, &ai, &aj, &a);

    //Read seed
    filename = "../../graph/minnesota_R.smat";
    stringstream ss;
    int64_t nR = 0;
    int64_t* R = NULL;
    read_seed<int64_t, int64_t>(filename.c_str(), &nR, &R);
	int64_t* ret_set = (int64_t*)malloc(sizeof(int64_t)*nR);

    //Begin calling C function
    cout << "calling C function" << endl;
    int64_t actual_length = MQI64(m, nR, ai, aj, 0, R, ret_set);
    cout << "output set" << endl;
    for(int i = 0; i < actual_length; i ++){
        cout << ret_set[i] << " ";
    }
    cout << endl << "total number of vertices is " << actual_length << endl;
    free(R);
    free(ai);
    free(aj);
    free(a);

    //Check the output
    cout << "compare the output with correct results" << endl;
    filename = "correct_output/MQI/minnesota_results.smat";
    char* read_file = readSMAT(filename.c_str());
    ss << read_file;
    free(read_file);
    int64_t correct_length;
    ss >> correct_length;
    int64_t* correct_ret_set = (int64_t *)malloc(sizeof(int64_t) * correct_length);
    for(size_t i = 0; i < correct_length; i ++){
        ss >> correct_ret_set[i];
    }
    ss.str("");
    if(actual_length != correct_length){
        cout << "output length is not correct!" << endl;
        return EXIT_FAILURE;
    }
    else{
        for(size_t i = 0; i < correct_length; i ++){
            if(ret_set[i] != correct_ret_set[i]){
                cout << "output is not correct!" << endl;
                return EXIT_FAILURE;
            }
        }
    }
    cout << "output is correct!" << endl;
    free(correct_ret_set);
    free(ret_set);
	return EXIT_SUCCESS;
}
