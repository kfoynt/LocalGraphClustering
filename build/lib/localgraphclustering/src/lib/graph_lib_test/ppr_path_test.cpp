#include <stdlib.h>
#include <iostream>
#include "../include/ppr_path.hpp"
#include "../include/ppr_path_c_interface.h"
#include <stdint.h>
#include <sstream>
#include <string>
#include "../include/readData.hpp"
#include <cmath>

using namespace std;

int main()
{
    cout << "test ppr_path on file usps_3nn.smat with 0 offset" << endl;

    //Read and convert data
    string filename;
    filename = "../../graph/usps_3nn.smat";
    int64_t m = 0, n = 0;
    int64_t* ai = NULL, *aj = NULL;
    double* a = NULL;
    read_and_convert<int64_t, int64_t>(filename.c_str(), &m, &n, &ai, &aj, &a);
    
    //Read seed
    filename = "../../graph/usps_3nn_seed.smat";
    stringstream ss;
    int64_t nseedids = 0;
    int64_t* seedids = NULL;
    read_seed<int64_t, int64_t>(filename.c_str(), &nseedids, &seedids);
	
    //define parameters and allocate enough memory
    double alpha = 0.99;
	double eps = 0.0001;
    int64_t max_step = (int64_t)1 / ((1 - alpha) * eps);
	int64_t* xids = (int64_t*)malloc(sizeof(int64_t)*m);
    int64_t num_eps = 0;
    double* epsilon = (double*)malloc(sizeof(double) * max_step);
    double* conds = (double*)malloc(sizeof(double) * max_step);
    double* cuts = (double*)malloc(sizeof(double) * max_step);
    double* vols = (double*)malloc(sizeof(double) * max_step);
    int64_t* setsizes = (int64_t*)malloc(sizeof(int64_t) * max_step);
    int64_t* stepnums = (int64_t*)malloc(sizeof(int64_t) * max_step);
    struct path_info ret_path_results = {.num_eps = &num_eps, .epsilon = epsilon,
        .conds = conds, .cuts = cuts, .vols = vols, .setsizes = setsizes, .stepnums = stepnums};
    
    int64_t nrank_changes = 0, nrank_inserts = 0, nsteps = 0, size_for_best_cond = 0;
    int64_t* starts = (int64_t*)malloc(sizeof(int64_t) * max_step);
    int64_t* ends = (int64_t*)malloc(sizeof(int64_t) * max_step);
    int64_t* nodes = (int64_t*)malloc(sizeof(int64_t) * max_step);
    int64_t* deg_of_pushed = (int64_t*)malloc(sizeof(int64_t) * max_step);
    int64_t* size_of_solvec = (int64_t*)malloc(sizeof(int64_t) * max_step);
    int64_t* size_of_r = (int64_t*)malloc(sizeof(int64_t) * max_step);
    double* val_of_push = (double*)malloc(sizeof(double) * max_step);
    double* global_bcond = (double*)malloc(sizeof(double) * max_step);
    struct rank_info ret_rank_results = {.starts = starts, .ends = ends, .nodes = nodes,
        .deg_of_pushed = deg_of_pushed, .size_of_solvec = size_of_solvec, .size_of_r = size_of_r,
        .val_of_push = val_of_push, .global_bcond = global_bcond, .nrank_changes = &nrank_changes,
        .nrank_inserts = &nrank_inserts, .nsteps = &nsteps, .size_for_best_cond = &size_for_best_cond};

    //Begin calling C function
    cout << "calling C function" << endl;
	int64_t actual_length = ppr_path64(m,ai,aj,0,alpha,eps,0.1,seedids,nseedids,xids,m,ret_path_results,ret_rank_results);

	cout << "actual length" << endl;
    cout<<actual_length<<endl;
    cout << "best set" << endl;
    for(size_t i = 0; i < (size_t)actual_length; ++ i){
        cout << xids[i] << " ";
    }
    cout << endl;
    free(ai);
    free(aj);
    free(a);

    //Check the output
    cout << "compare the output with correct results" << endl;
    filename = "correct_output/ppr_path/usps_3nn_bestclus.smat";
    char* read_file = readSMAT(filename.c_str());
    ss << read_file;
    free(read_file);
    int64_t correct_length;
    ss >> correct_length;
    int64_t* correct_xids = (int64_t *)malloc(sizeof(int64_t) * correct_length);
    for(size_t i = 0; i < (size_t)correct_length; i ++){
        ss >> correct_xids[i];
    }
    ss.str("");
    
    cout << "check best cluster" << endl;
    if(actual_length != correct_length){
        cout << "best cluster length is not correct!" << endl;
        return EXIT_FAILURE;
    }
    else{
        for(size_t i = 0; i < (size_t)correct_length; i ++){
            if(xids[i] != correct_xids[i]){
                cout << "best cluster is not correct!" << endl;
                return EXIT_FAILURE;
            }
        }
    }
    cout << "best cluster is correct!" << endl;
    free(correct_xids);
    free(xids);
    
    
    filename = "correct_output/ppr_path/usps_3nn_eps_stats.smat";
    read_file = readSMAT(filename.c_str());
    ss << read_file;
    free(read_file);
    int64_t correct_num_eps;
    ss >> correct_num_eps;
    cout << "check eps stats" << endl;
    if(num_eps != correct_num_eps){
        cout << "eps stats length is not correct!" << endl;
        return EXIT_FAILURE;
    }
    double* correct_epsilon = (double*)malloc(sizeof(double) * num_eps);
    double* correct_conds = (double*)malloc(sizeof(double) * num_eps);
    double* correct_cuts = (double*)malloc(sizeof(double) * num_eps);
    double* correct_vols = (double*)malloc(sizeof(double) * num_eps);
    int64_t* correct_setsizes = (int64_t*)malloc(sizeof(int64_t) * num_eps);
    int64_t* correct_stepnums = (int64_t*)malloc(sizeof(int64_t) * num_eps);
    for(size_t i = 0; i < (size_t)num_eps; i ++){
        ss >> correct_epsilon[i];
        ss >> correct_conds[i];
        ss >> correct_cuts[i];
        ss >> correct_vols[i];
        ss >> correct_setsizes[i];
        ss >> correct_stepnums[i];
    }
    for(size_t i = 0; i < (size_t)num_eps; i ++){
        if(abs(epsilon[i] - correct_epsilon[i]) > pow(10,-5)
           || abs(conds[i] - correct_conds[i]) > pow(10,5)
           || abs(cuts[i] - correct_cuts[i]) > pow(10,-5)
           || abs(vols[i] - correct_vols[i]) > pow(10,-5)
           || setsizes[i] != correct_setsizes[i] || stepnums[i] != correct_stepnums[i]){
            cout << "eps stats is not correct!" << endl;
            return EXIT_FAILURE;
        }
    }
    cout << "eps stats is correct!" << endl;
    free(correct_epsilon);
    free(correct_conds);
    free(correct_cuts);
    free(correct_vols);
    free(correct_setsizes);
    free(correct_stepnums);
    ss.str("");
    
    
    filename = "correct_output/ppr_path/usps_3nn_rank_stats.smat";
    read_file = readSMAT(filename.c_str());
    ss << read_file;
    free(read_file);
    int64_t correct_nrank_changes, correct_nrank_inserts, correct_nsteps, correct_size_for_best_cond;
    ss >> correct_nrank_changes;
    ss >> correct_nrank_inserts;
    ss >> correct_nsteps;
    ss >> correct_size_for_best_cond;
    cout << "check rank stats" << endl;
    if(nrank_changes != correct_nrank_changes || nrank_inserts != correct_nrank_inserts
       || nsteps != correct_nsteps || size_for_best_cond != correct_size_for_best_cond){
        cout << "rank stats length is not correct!" << endl;
        cout << nrank_changes << endl;
        cout << nrank_inserts << endl;
        cout << nsteps << endl;
        cout << size_for_best_cond << endl;
        return EXIT_FAILURE;
    }
    int64_t* correct_starts = (int64_t*)malloc(sizeof(int64_t) * nsteps);
    int64_t* correct_ends = (int64_t*)malloc(sizeof(int64_t) * nsteps);
    int64_t* correct_nodes = (int64_t*)malloc(sizeof(int64_t) * nsteps);
    int64_t* correct_deg_of_pushed = (int64_t*)malloc(sizeof(int64_t) * nsteps);
    int64_t* correct_size_of_solvec = (int64_t*)malloc(sizeof(int64_t) * nsteps);
    int64_t* correct_size_of_r = (int64_t*)malloc(sizeof(int64_t) * nsteps);
    double* correct_val_of_push = (double*)malloc(sizeof(double) * nsteps);
    double* correct_global_bcond = (double*)malloc(sizeof(double) * nsteps);
    for(size_t i = 0; i < (size_t)nsteps; i ++){
        ss >> correct_starts[i];
        ss >> correct_ends[i];
        ss >> correct_nodes[i];
        ss >> correct_deg_of_pushed[i];
        ss >> correct_size_of_solvec[i];
        ss >> correct_size_of_r[i];
        ss >> correct_val_of_push[i];
        ss >> correct_global_bcond[i];
    }
    for(size_t i = 0; i < (size_t)nsteps; i ++){
        if(starts[i] != correct_starts[i] || ends[i] != correct_ends[i] || nodes[i] != correct_nodes[i]
           || deg_of_pushed[i] != correct_deg_of_pushed[i] || size_of_solvec[i] != correct_size_of_solvec[i]
           || size_of_r[i] != correct_size_of_r[i] || abs(val_of_push[i] - correct_val_of_push[i]) > pow(10,-5)
           || abs(global_bcond[i] - correct_global_bcond[i]) > pow(10,5)){
            cout << "rank stats is not correct!" << endl;
            return EXIT_FAILURE;
        }
    }
    cout << "rank stats is correct!" << endl;
    free(correct_starts);
    free(correct_ends);
    free(correct_nodes);
    free(correct_deg_of_pushed);
    free(correct_size_of_solvec);
    free(correct_size_of_r);
    free(correct_val_of_push);
    free(correct_global_bcond);
    ss.str("");

    cout << "output is correct!" << endl;
	return EXIT_SUCCESS;
}
