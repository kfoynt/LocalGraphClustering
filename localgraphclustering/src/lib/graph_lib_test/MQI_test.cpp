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
    string filename1;
    filename1 = "../../graph/minnesota.smat";
    int64_t m1 = 0, n1 = 0;
    int64_t* ai1 = NULL, *aj1 = NULL;
    double* a1 = NULL;
    read_and_convert<int64_t, int64_t>(filename1.c_str(), &m1, &n1, &ai1, &aj1, &a1);

    //Read seed
    int64_t nR1 = 216;
    int64_t R1[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,73,74,75,76,77,78,79,80,81,82,83,84,85,87,88,89,90,91,92,93,94,95,97,98,99,100,102,103,104,105,106,108,112,114,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,139,140,141,142,143,144,145,147,148,149,150,151,152,155,157,158,159,160,161,162,164,165,166,168,169,171,172,173,176,177,178,179,180,185,187,188,191,192,195,196,197,201,208,209,210,211,212,215,217,218,219,221,223,225,226,227,228,231,232,244,245,246,247,248,249,253,254,257,261,262,265,269,270,271,272,273,275,276,277,278,279,285,286,287,290,291,299,303,323,327};
    int64_t* ret_set1 = (int64_t*)malloc(sizeof(int64_t)*nR1);

    //Begin calling C function
    cout << "calling C function" << endl;
    int64_t actual_length1 = MQI64(m1, nR1, ai1, aj1, 0, R1, ret_set1);
    cout << "output set" << endl;
    for(int i = 0; i < actual_length1; i ++){
        cout << ret_set1[i] << " ";
    }
    cout << endl << "total number of vertices is " << actual_length1 << endl;
    free(ai1);
    free(aj1);
    free(a1);

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
	return EXIT_SUCCESS;

    
}
