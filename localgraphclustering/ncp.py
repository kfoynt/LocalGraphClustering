from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
import pandas as pd

import time
from collections import namedtuple
import threading
import math
import warnings

from localgraphclustering.simple_interface import * 

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
    

def ncp_node_worker(ncpdata,sets,func,timeout_ncp):
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
def ncp_neighborhood_worker(ncpdata,sets,func,timeout_ncp):
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
def ncp_set_worker(ncpdata,setnos,func,timeout_ncp):
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
    def __init__(self, graph, setfuncs=[],input_stats=True,do_largest_component=True):
        if do_largest_component:
            self.graph = graph.largest_component()
        else:
            self.graph = graph
            
        # Todo - have "largest_component" return a graph for the largest component
        self.input_stats = input_stats
        self.set_funcs = setfuncs        
        self.results = []

        standard_fields = self.graph.set_scores([0])
        for F in self.set_funcs: # build the list of keys for set_funcs
            standard_fields.update(F(self.graph, [0]))
        if self.input_stats:
            result_fields = ["input_" + field for field in standard_fields.keys()]
        result_fields.extend(["output_" + str(field) for field in standard_fields.keys()])
        result_fields.extend(["methodfunc", "input_set_type", "input_set_params", "time"])
        self.record = namedtuple("NCPDataRecord", field_names=result_fields)
        self.default_method = None
        self.sets = []
        self.method_names = {} # This stores human readable and usable names for methods
        
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
        
        for i in range(nthreads):
            sets = [ [j] for j in threadnodes[i] ] # make a set of sets
            t = threading.Thread(target=ncp_node_worker,args=(self, sets, method, timeout))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
    def add_random_neighborhood_samples(self, ratio=0.3, timeout=1000, nthreads=4, method=None, methodname=None):
        method = self._check_method(method, methodname)
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
            
    def add_set_samples(self, sets, nthreads=4, method=None, methodname=None, timeout=1000):
        method = self._check_method(method, methodname)
        threads = []
        startset = len(self.sets)
        self.sets.extend(sets)
        endset = len(self.sets)
        
        setnos = np.array_split(range(startset,endset), nthreads) # set numbers
        
        for i in range(nthreads):
            t = threading.Thread(target=ncp_set_worker,args=(self, setnos[i], method, timeout))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        
    def as_data_frame(self):
        df = pd.DataFrame.from_records(self.results, columns=self.record._fields)
        df.rename(columns={'method':'methodfunc'}, inplace=True)
        df["method"] = df["methodfunc"].map(self.method_names)
        return df


Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=pd.DataFrame)

class Ncp(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the Ncp class.
        """

        super().__init__()

    def produce(self, 
                input: Input, 
                method: str,
                ratio: float = 0.3,
                timeout: float = 1000,
                nthreads: int = 4,
                do_largest_component: bool = True,
                U: int = 3,
                h: int = 10,
                w: int = 2,
                gamma: float = 0.01/0.99,
                epsilon: float = 1.0e-1,
                rholist: List[float] = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4],
                ) -> Output:
        """
        Network Community Profile for the largest connected components of the graph. For details please refer to: 
        Jure Leskovec, Kevin J Lang, Anirban Dasgupta, Michael W Mahoney. Community structure in 
        large networks: Natural cluster sizes and the absence of large well-defined clusters.
        The NCP is computed for the largest connected component of the given graph.

        Parameters
        ----------  
        
        input: Graph
            Given graph whose network community Profile needs to be computed.

        method: str
            Choose either Capacity Releasing Diffusion or Max Flow Quotient Cut Improvement as the local clustering
            method, must be "crd", "mqi", "l1reg" or "approxPageRank".

        ratio: float
            Ratio of nodes to be used for computation of NCP.
            It should be between 0 and 1.

        Parameters (optional)
        ---------------------

        timeout: float
            default = 1000
            Maximum time in seconds for NCP calculation.

        nthreads: int
            default = 4
            Choose the number of threads used for NCP calculation
            
        do_largest_component: bool
            default = True
            If true it computes the NCP for the largest connected component.
            This task might double the required memory. If False then it computes
            the NCP for the given graph. This taks has minimal memory requirements.

        U: integer
            default == 3
            The net mass any edge can be at most.
          
        h: integer
            default == 10
            The label of any node can have at most.
          
        w: integer
            default == 2
            Multiplicative factor for increasing the capacity of the nodes at each iteration.
            
        gamma: float
            default == 0.01/0.99 
            This is the value of gamma to use in the approximate PageRank or l1 regularized diffusion.
        
        rholist: list-of-float
            default == [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4]
            For l1reg and approx_pagerank, we have a variety of regularization
            levels that are tested. This parameter gives the list. 

        epsilon: float
            default == 0.1
            Tolerance for FISTA for solving the l1-regularized personalized PageRank problem.
          
        Returns
        -------

        df: pandas.DataFrame
            The output can be used as the parameter of NCPPlots to plot all sorts of graph.
        """ 
        G = input
        G.compute_statistics()
        ncp = NCPData(G,do_largest_component=do_largest_component)
        
        if method == "crd":
            func = lambda G,R: simple_crd(G,R,w=w, U=U, h=h)
            ncp.add_random_neighborhood_samples(method=func,methodname="crd",
                ratio=ratio,nthreads=nthreads,timeout=timeout/2)
            ncp.add_random_node_samples(method=func,methodname="crd",
                ratio=ratio,nthreads=nthreads,timeout=timeout/2)
        elif method == "mqi":
            func = lambda G,R: simple_mqi(G,R)
            ncp.add_random_neighborhood_samples(ratio=ratio,nthreads=nthreads,timeout=timeout,
                method=func,methodname="mqi")
        elif method == "l1reg":
            # convert [0,inf] to [0,1] where 0 -> 0 (no localization) and inf -> 1 (all prior)
            beta = 1.0-1.0/(1.0+gamma) 
            funcs = {lambda G,R: simple_l1_diffusion(G,R,alpha=beta,rho=rho,epsilon=epsilon):'l1reg;rho=%.0e'%(rho) 
                            for rho in rholist}
            for func in funcs.keys():
                ncp.add_random_node_samples(method=func,methodname=funcs[func],
                    ratio=ratio,nthreads=nthreads,timeout=timeout/len(funcs))                    
        elif method == "approxPageRank":  
            beta = 1.0-1.0/(1.0+gamma) 
            funcs = {lambda G,R: simple_approx_pagerank(G,R,alpha=beta,rho=rho):'approxPageRank;rho=%.0e'%(rho) 
                            for rho in rholist}
            for func in funcs.keys():
                ncp.add_random_node_samples(method=func,methodname=funcs[func],
                    ratio=ratio,nthreads=nthreads,timeout=timeout/len(funcs))  
        else:
            raise(ValueError("Must specify a method (crd, mqi or l1reg)."))
        
        return ncp
        
