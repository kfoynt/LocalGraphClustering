#include <stdlib.h>
#include <iostream>
#include "../include/sweepcut_c_interface.h"
#include "../include/readData.hpp"

using namespace std;

int main()
{
    cout << "test sweepcut on file minnesota.smat with 0 offset" << endl;

    //Read and convert data
    string filename;
    filename = "../../graph/minnesota.smat";
    int64_t m = 0, n = 0;
    int64_t* ai = NULL, *aj = NULL;
    double* a = NULL;
    for(int i = 0; i < n; i ++){
        a[i] = 1.0;
    }
    read_and_convert<int64_t, int64_t>(filename.c_str(), &m, &n, &ai, &aj, &a);

    //Read seed
    filename = "../../graph/minnesota_ids.smat";
    stringstream ss;
    int64_t nids = 0;
    int64_t* ids = NULL;
    read_seed<int64_t, int64_t>(filename.c_str(), &nids, &ids);
    int64_t* bestclus = (int64_t*)malloc(sizeof(int64_t) * nids);
    
    //Begin calling C function
    cout << "calling C function" << endl;
    int64_t offset = 0;
    double ret_cond = 0.0;
    int64_t actual_length = sweepcut_without_sorting64(ids, bestclus, nids,
                                                       m, ai, aj, a, offset, &ret_cond, NULL);
    cout << "actual length" << endl << actual_length << endl;
    cout << "min conductance" << endl << ret_cond << endl;
    cout << "best set" << endl;
    for(int i = 0; i < actual_length; i ++)
    {
        cout << bestclus[i] << " ";
    }
    cout << endl;
    free(ids);
    free(ai);
    free(aj);
    free(a);

    //Check the output
    cout << "compare the output with correct results" << endl;
    filename = "correct_output/sweepcut/minnesota_results.smat";
    char* read_file = readSMAT(filename.c_str());
    ss << read_file;
    free(read_file);
    int64_t correct_length;
    ss >> correct_length;
    int64_t* correct_clus = (int64_t *)malloc(sizeof(int64_t) * correct_length);
    for(size_t i = 0; i < (size_t)correct_length; i ++){
        ss >> correct_clus[i];
    }
    ss.str("");
    if(actual_length != correct_length){
        cout << "output length is not correct!" << endl;
        return EXIT_FAILURE;
    }
    else{
        for(size_t i = 0; i < (size_t)correct_length; i ++){
            if(bestclus[i] != correct_clus[i]){
                cout << "output is not correct!" << endl;
                return EXIT_FAILURE;
            }
        }
    }
    cout << "output is correct!" << endl;
    free(correct_clus);
    free(bestclus);
	return EXIT_SUCCESS;
}
