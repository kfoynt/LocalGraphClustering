from typing import *
import numpy as np
import pandas as pd

import time
from collections import namedtuple
import threading
import math
import warnings

from .spectral_clustering import spectral_clustering
from .flow_clustering import flow_clustering
from .GraphLocal import GraphLocal

def ncp_experiment(ncpdata,R,func,method_stats):
    if ncpdata.input_stats:
        input_stats = ncpdata.graph.set_scores(R)
        for F in ncpdata.set_funcs: # build the list of keys for set_funcs
            input_stats.update(F(ncpdata.graph, R))
        input_stats = {"input_" + str(key):value for key,value in input_stats.items() } # add input prefix
    else:
        input_stats = {}
        
    start = time.time()
    S = func(ncpdata.graph, R)
    dt = time.time() - start
    
    if len(S) > 0:
        output_stats = ncpdata.graph.set_scores(S)
        for F in ncpdata.set_funcs: # build the list of keys for set_funcs
            output_stats.update(F(ncpdata.graph, S))
        output_stats = {"output_" + str(key):value for key,value in output_stats.items() } # add output prefix

        method_stats['methodfunc']  = func
        method_stats['time'] = dt
        return [ncpdata.record(**input_stats, **output_stats, **method_stats)]
    else:
        return [] # nothing to return
    

def ncp_node_worker(ncpdata,sets,func,timeout_ncp,methodname):
    start = time.time()
    setno = 0
    for R in sets:
        #print("setno = %i"%(setno))
        setno += 1
        
        method_stats = {'input_set_type': 'node', 'input_set_params':R[0]}
        ncpdata.results.extend(ncp_experiment(ncpdata, R, func, method_stats))
        
        end = time.time()
        if end - start > timeout_ncp:
            break

# todo avoid code duplication 
def ncp_neighborhood_worker(ncpdata,sets,func,timeout_ncp,methodname):
    start = time.time()
    setno = 0
    for R in sets:
        #print("setno = %i"%(setno))
        setno += 1
        
        R = R.copy() # duplicate so we don't keep extra weird data around
        node = R[0]
        R.extend(ncpdata.graph.neighbors(R[0]))
        method_stats = {'input_set_type': 'neighborhood', 'input_set_params':node}
        
        ncpdata.results.extend(ncp_experiment(ncpdata, R, func, method_stats))
        
        end = time.time()
        if end - start > timeout_ncp:
            break          
            
# todo avoid code duplication 
def ncp_set_worker(ncpdata,setnos,func,timeout_ncp,methodname):
    start = time.time()
    setno = 0
    for id in setnos:
        #print("setno = %i"%(setno))
        setno += 1
        R = ncpdata.sets[id]
        #R = R.copy() # duplicate so we don't keep extra weird data around
        method_stats = {'input_set_type': 'set', 'input_set_params':id}
        
        ncpdata.results.extend(ncp_experiment(ncpdata, R, func, method_stats))
        
        end = time.time()
        if end - start > timeout_ncp:
            break 

class NCPData:    
    def __init__(self, graph, setfuncs=[], input_stats=True, do_largest_component=True):
        if do_largest_component:
            self.graph = graph.largest_component()
        else:
            self.graph = graph
            
        # Todo - have "largest_component" return a graph for the largest component
        self.input_stats = input_stats
        self.set_funcs = setfuncs        

        standard_fields = self.graph.set_scores([0])
        for F in self.set_funcs: # build the list of keys for set_funcs
            standard_fields.update(F(self.graph, [0]))
        if self.input_stats:
            result_fields = ["input_" + field for field in standard_fields.keys()]
        result_fields.extend(["output_" + str(field) for field in standard_fields.keys()])
        result_fields.extend(["methodfunc", "input_set_type", "input_set_params", "time"])
        self.record = namedtuple("NCPDataRecord", field_names=result_fields)
        self.reset_records()
        self.default_method = None
        self.method_names = {} # This stores human readable and usable names for methods

    def reset_records(self):
        self.results = []
        self.sets = []
        self.method_names = {}
        
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
    
    def _check_method(self, method, name):
        if method == None:
            method = self.default_method
            if method == None:
                raise(ValueError("NCP call with method=None, but no default method specified"))
        if method in self.method_names:
            # in this case, we already have a name...
            if name is not None and name is not self.method_names[method]:
                warnings.warn("Duplicate name given to NCP method\n" +
                              "original name was: %s\n"%(self.method_names[method]) + 
                              "new name attempted: %s\n"%(name), stacklevel=3)
        else:
            # we need to name the method
            if name is None:
                # need to figure out a name, let's use something boring.
                self.method_names[method] = "method-%i"%(len(self.method_names))
            else:
                self.method_names[method] = name
            
        return method
        
    def add_random_node_samples(self, ratio=0.3, timeout=1000, nthreads=4, method=None, methodname=None):
        method = self._check_method(method, methodname)
        nodes = self.random_nodes(ratio)
        
        threads = []
        threadnodes = np.array_split(nodes, nthreads)
        
        """
        results_id = len(self.results)
        self.results.append([])
        self.method_names.append(methodname)
        """

        #self.results[methodname] = []
        
        for i in range(nthreads):
            sets = [ [j] for j in threadnodes[i] ] # make a set of sets
            t = threading.Thread(target=ncp_node_worker,args=(self, sets, method, timeout, methodname))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
    def add_random_neighborhood_samples(self, ratio=0.3, timeout=1000, nthreads=4, method=None, methodname=None):
        method = self._check_method(method, methodname)
        nodes = self.random_nodes(ratio)
        
        threads = []
        threadnodes = np.array_split(nodes, nthreads)
        
        """
        results_id = len(self.results)
        self.results.append([])
        self.method_names.append(methodname)
        """

        #self.results[methodname] = []
        
        for i in range(nthreads):
            sets = [ [j] for j in threadnodes[i] ] # make a set of sets
            t = threading.Thread(target=ncp_neighborhood_worker,args=(self, sets, method, timeout, methodname))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
    def add_set_samples(self, sets, nthreads=4, method=None, methodname=None, timeout=1000):
        method = self._check_method(method, methodname)
        threads = []
        startset = len(self.sets)
        self.sets.extend(sets)
        endset = len(self.sets)
        
        setnos = np.array_split(range(startset,endset), nthreads) # set numbers
        
        """
        results_id = len(self.results)
        self.results.append([])
        self.method_names.append(methodname)
        """

        #self.results[methodname] = []
        
        for i in range(nthreads):
            t = threading.Thread(target=ncp_set_worker,args=(self, setnos[i], method, timeout, methodname))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        
    def as_data_frame(self):
        """
        DF = pd.DataFrame.from_records([], columns=self.record._fields)
        for methodname in self.results.keys():
            df = pd.DataFrame.from_records(self.results[methodname], columns=self.record._fields)
            df.rename(columns={'method':'methodfunc'}, inplace=True)
            df["method"] = methodname
            DF = DF.append(df,ignore_index=True)
        return DF
        """
        df = pd.DataFrame.from_records(self.results, columns=self.record._fields)
        df.rename(columns={'method':'methodfunc'}, inplace=True)
        df["method"] = df["methodfunc"].map(self.method_names)
        return df

    def approxPageRank(self,
                       gamma: float = 0.01/0.99,
                       rholist: List[float] = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4],
                       ratio: float = 0.3,
                       nthreads: int = 4,
                       timeout: float = 1000):
        #self.reset_records("approxPageRank")
        alpha = 1.0-1.0/(1.0+gamma)
        funcs = {lambda G,R: spectral_clustering(G,R,alpha=alpha,rho=rho,method="acl")[0]:'acl;rho=%.0e'%(rho) 
                    for rho in rholist}
        for func in funcs.keys():
            self.add_random_node_samples(method=func,methodname=funcs[func],ratio=ratio,nthreads=nthreads,timeout=timeout/len(funcs))

    def l1reg(self,
              gamma: float = 0.01/0.99,
              rholist: List[float] = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4],
              ratio: float = 0.3,
              nthreads: int = 4,
              timeout: float = 1000):
        #self.reset_records("l1reg")
        alpha = 1.0-1.0/(1.0+gamma)
        funcs = {lambda G,R: spectral_clustering(G,R,alpha=alpha,rho=rho,method="l1reg")[0]:'l1reg;rho=%.0e'%(rho) 
                    for rho in rholist}
        for func in funcs.keys():
            self.add_random_node_samples(method=func,methodname=funcs[func],ratio=ratio,nthreads=nthreads,timeout=timeout/len(funcs))

    def crd(self,
            U: int = 3,
            h: int = 10,
            w: int = 2,
            ratio: float = 0.3,
            nthreads: int = 4,
            timeout: float = 1000):
        #self.reset_records("crd")
        func = lambda G,R: flow_clustering(G,R,w=w, U=U, h=h,method="crd")[0]
        self.add_random_neighborhood_samples(method=func,methodname="crd",
                ratio=ratio,nthreads=nthreads,timeout=timeout/2)
        self.add_random_node_samples(method=func,methodname="crd",
                ratio=ratio,nthreads=nthreads,timeout=timeout/2)
        return self

    def mqi(self,
            ratio: float = 0.3,
            nthreads: int = 4,
            timeout: float = 1000):
        #self.reset_records("mqi")
        func = lambda G,R: flow_clustering(G,R,method="mqi")[0]
        self.add_random_neighborhood_samples(ratio=ratio,nthreads=nthreads,timeout=timeout,
                method=func,methodname="mqi")
        
