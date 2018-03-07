import numpy as np
import pandas as pd
import time
from collections import namedtuple
import threading
import math

from localgraphclustering.NCPAlgo import * 

class NCPData:    
    def __init__(self, graph, setfuncs=[],input_stats=True,do_largest_component=True):
        if do_largest_component:
            self.graph = graph.largest_component()
        else:
            self.graph = graph
            
        # Todo - have "largest_component" return a graph for the largest component
        self.input_stats = input_stats
        self.set_funcs = setfuncs        
        self.results = []

        standard_fields = graph_set_scores(self.graph, [0])
        for F in self.set_funcs: # build the list of keys for set_funcs
            standard_fields.update(F(self.graph, [0]))
        if self.input_stats:
            result_fields = ["input_" + field for field in standard_fields.keys()]
        result_fields.extend(["output_" + str(field) for field in standard_fields.keys()])
        result_fields.extend(["methodfunc", "input_set_type", "input_set_params", "time"])
        self.record = namedtuple("NCPDataRecord", field_names=result_fields)
        self.default_method = None
        self.sets = []
        
    def random_nodes(self, ratio):
        n = self.graph._num_vertices
        if 0 < ratio <= 1: 
            n_nodes = min(np.ceil(ratio*n),n)
            n_nodes = int(n_nodes)
        elif ratio > 1.0: # this is a number of nodes
            n_nodes = int(ratio) 
        else:
            raise(ValueError("the ratio parameter must be positive"))
            
        return np.random.choice(np.arange(0,n), size=n_nodes, replace=False)
        
    """ Decode and return the input set to a particular experiment. """
    def input_set(self, j): 
        result = self.results[j] # todo check for validity
        if result.input_set_type=="node":
            return [result.input_set_params]
        elif result.input_set_type=="neighborhood":
            R = self.graph.neighbors(result.input_set_params)
            R.append(result.input_set_params)
            return R
        elif result.input_set_type=="set":
            return self.sets[result.input_set_params].copy()
        else: 
            raise(ValueError("the input_set_type is unrecognized"))
        
    def output_set(self, j):
        result = self.results[j] # todo check for validity
        func = result.methodfunc
        R = self.input_set(j)
        return func(self.graph, R)
    
    def _check_method(self, method):
        if method == None:
            method = self.default_method
            if method == None:
                raise(ValueError("NCP call with method=None, but no default method specified"))
        return method
        
    def add_random_node_samples(self, ratio=0.3, timeout=1000, nthreads=4, method=None):
        method = self._check_method(method)
        nodes = self.random_nodes(ratio)
        
        threads = []
        threadnodes = np.array_split(nodes, nthreads)
        
        for i in range(nthreads):
            sets = [ [j] for j in threadnodes[i] ] # make a set of sets
            t = threading.Thread(target=ncp_node_worker,args=(self, sets, method, timeout))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
    def add_random_neighborhood_samples(self, ratio=0.3, timeout=1000, nthreads=4, method=None):
        method = self._check_method(method)
        nodes = self.random_nodes(ratio)
        
        threads = []
        threadnodes = np.array_split(nodes, nthreads)
        
        for i in range(nthreads):
            sets = [ [j] for j in threadnodes[i] ] # make a set of sets
            t = threading.Thread(target=ncp_neighborhood_worker,args=(self, sets, method, timeout))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
    def add_set_samples(self, sets, nthreads=4, method=None):
        method = self._check_method(method)
        threads = []
        startset = len(ncp.sets)
        self.sets.extend(sets)
        endset = len(ncp.sets)
        
        setnos = np.array_split(range(startset,endset)) # set numbers
        
        for i in range(nthreads):
            t = threading.Thread(target=ncp_set_worker,args=(self, setnos[i], method, timeout))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        
    def as_data_frame(self):
        return pd.DataFrame.from_records(self.results, columns=self.record._fields)