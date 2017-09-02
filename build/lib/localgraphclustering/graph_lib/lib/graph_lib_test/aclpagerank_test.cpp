#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "../include/aclpagerank_c_interface.h"
#include "../include/readData.hpp"

using namespace std;

int main()
{
    cout << "test aclpagerank on file Unknown.smat with 0 offset" << endl;

    //Read and convert data
    string filename;
    filename = "../../graph/Unknown.smat";
    int64_t m = 0, n = 0;
    int64_t* ai = NULL, *aj = NULL;
    double* a = NULL;
    read_and_convert<int64_t, int64_t>(filename.c_str(), &m, &n, &ai, &aj, &a);
    
    //Read seed
    filename = "../../graph/Unknown_seed.smat";
    stringstream ss;
    int64_t nseedids = 0;
    int64_t* seedids = NULL;
    read_seed<int64_t, int64_t>(filename.c_str(), &nseedids, &seedids);
	double alpha = 0.99;
	double eps = pow(10,-7);
    int64_t xlength = 100;
    int64_t maxstep = (size_t)1/(eps*(1-alpha));
	int64_t* xids = (int64_t*)malloc(sizeof(int64_t)*m);
	double* values = (double*)malloc(sizeof(double)*m);

    //Begin calling C function
    cout << "calling C function " << endl;
	int64_t actual_length = aclpagerank64(m,ai,aj,0,alpha,eps,seedids,
            nseedids,maxstep,xids,xlength,values);
	cout << "actual length" << endl;
    cout<<actual_length<<endl;
    cout << "nonzero pagerank sets and values" << endl;
    for(size_t i = 0; i < actual_length; ++ i){
        cout << xids[i] << " ";
    }
    cout << endl;
    for(size_t i = 0; i < actual_length; ++ i){
        cout << values[i] << " ";
    }
    cout << endl;
    free(ai);
    free(aj);
    free(a);
    free(seedids);

    //Check the output
    cout << "compare the output with correct results" << endl;
    filename = "correct_output/aclpagerank/Unknown_set.smat";
    char* read_file = readSMAT(filename.c_str());
    ss << read_file;
    free(read_file);
    int64_t correct_length;
    ss >> correct_length;
    int64_t* correct_xids = (int64_t *)malloc(sizeof(int64_t) * correct_length);
    for(size_t i = 0; i < correct_length; i ++){
        ss >> correct_xids[i];
    }
    ss.str("");
    filename = "correct_output/aclpagerank/Unknown_values.smat";
    read_file = readSMAT(filename.c_str());
    ss << read_file;
    free(read_file);
    ss >> correct_length;
    double* correct_values = (double *)malloc(sizeof(double) * correct_length);
    for(size_t i = 0; i < correct_length; i ++){
        ss >> correct_values[i];
    }
    ss.str("");

    if(actual_length != correct_length){
        cout << "output length is not correct!" << endl;
        return EXIT_FAILURE;
    }
    else{
        for(size_t i = 0; i < correct_length; i ++){
            if(xids[i] != correct_xids[i] || fabs(values[i] - correct_values[i]) > pow(10, -5)){
                cout << "output is not correct!" << endl;
                return EXIT_FAILURE;
            }
        }
    }
    cout << "output is correct!" << endl;
    free(correct_xids);
    free(correct_values);
    free(values);
    free(xids);
	return EXIT_SUCCESS;
}
