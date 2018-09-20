from typing import *
import numpy as np
import pandas as pd

import time
from collections import namedtuple
import threading
import math
import warnings
import copy
import multiprocessing as mp
import functools

from .spectral_clustering import spectral_clustering
from .flow_clustering import flow_clustering
from .GraphLocal import GraphLocal
from .triangleclusters import triangleclusters
from .cpp import *

def _partial_functions_equal(func1, func2):
    if not (isinstance(func1, functools.partial) and isinstance(func2, functools.partial)):
        return False
    are_equal = all([getattr(func1, attr) == getattr(func2, attr) for attr in ['func', 'args', 'keywords']])
    if are_equal == False:
        # TODO remove this code once we are sure things are working okay
        print(func1, "!=", func2)
        for attr in ['func', 'args', 'keywords']:
            print(getattr(func1, attr), getattr(func2, attr))
    return are_equal

def ncp_experiment(ncpdata,R,func,method_stats):
    if ncpdata.input_stats:
        input_stats = ncpdata.graph.set_scores(R)
        for F in ncpdata.set_funcs: # build the list of keys for set_funcs
            input_stats.update(F(ncpdata.graph, R))
        input_stats = {"input_" + str(key):value for key,value in input_stats.items() } # add input prefix
    else:
        input_stats = {}

    start = time.time()
    S = func(ncpdata.graph, R)[0]
    dt = time.time() - start

    if len(S) > 0:
        output_stats = ncpdata.graph.set_scores(S)
        for F in ncpdata.set_funcs: # build the list of keys for set_funcs
            output_stats.update(F(ncpdata.graph, S))
        output_stats = {"output_" + str(key):value for key,value in output_stats.items() } # add output prefix

        method_stats['methodfunc']  = func
        method_stats['time'] = dt
        return [ncpdata.record(**input_stats, **output_stats, **method_stats)._asdict()]
    else:
        return [] # nothing to return


""" This is how worker's get the graph data and the NCP setup information. """
# we are now using stuff from here
# https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
workvars = {}
def _ncp_worker_setup(ncpdata):
    ncp = copy.copy(ncpdata)
    ncp.graph = None # remove the graph
    ncp.results = []
    # TODO - Figure out some efficient way of copying over sets...
    svardata = {}
    svardata["ncpdata"] = ncp
    svardata["g"] = ncpdata.graph.to_shared()
    return svardata

def _ncp_worker_init(svardata):
    workvars["ncpdata"] = svardata["ncpdata"]
    workvars["ncpdata"].graph = GraphLocal.from_shared(svardata["g"])

def _ncp_node_worker(workid, setids,func,timeout):
    return ncp_worker(workid, "node", workvars["ncpdata"], setids, func, timeout)

def _ncp_neighborhood_worker(workid, setids,func,timeout):
    return ncp_worker(workid, "neighborhood", workvars["ncpdata"], setids, func, timeout)

def _ncp_set_worker(workid, setids,func,timeout):
    return ncp_worker(workid, "set", workvars["ncpdata"], setids, func, timeout)

def ncp_worker(workid, runtype, ncpdata, setids, func, timeout):
    start = time.time()
    setno = 0
    results = []
    for Rid in setids:
        setno += 1
        if runtype == 'node':
            method_stats = {'input_set_type': runtype, 'input_set_params':Rid[0]}
            result = ncp_experiment(ncpdata, Rid, func, method_stats)
            results.extend(result)
        elif runtype == 'neighborhood':
            R = Rid.copy() # duplicate so we don't keep extra weird data around
            node = R[0]
            R.extend(ncpdata.graph.neighbors(R[0]))
            method_stats = {'input_set_type': runtype, 'input_set_params':node}
            result = ncp_experiment(ncpdata, R, func, method_stats)
            results.extend(result)
        elif runtype == 'set':
            R = ncpdata.sets[Rid]
            method_stats = {'input_set_type': runtype, 'input_set_params':Rid}
            result = ncp_experiment(ncpdata, R, func, method_stats)
            results.extend(result)
        else:
            raise(ValueError("the runtype must be 'node','neighborhood', or 'set'"))

        end = time.time()
        if end - start > timeout:
            break
    return results

class NCPData:
    def __init__(self, graph, setfuncs=[], input_stats=True, do_largest_component=True):
        if do_largest_component:
            self.graph = graph.largest_component()
        else:
            self.graph = graph

        # Todo - have "largest_component" return a graph for the largest component
        self.input_stats = input_stats
        self.set_funcs = setfuncs
        self.nruns = 0

        standard_fields = self.graph.set_scores([0])
        for F in self.set_funcs: # build the list of keys for set_funcs
            standard_fields.update(F(self.graph, [0]))
        if self.input_stats:
            result_fields = ["input_" + field for field in standard_fields.keys()]
        result_fields.extend(["output_" + str(field) for field in standard_fields.keys()])
        result_fields.extend(["methodfunc", "input_set_type", "input_set_params", "time"])
        self.record = namedtuple("NCPDataRecord", field_names=result_fields)
        self.neighborhood_cond = None
        self.reset_records()
        self.default_method = None
        self.method_names = {} # This stores human readable and usable names for methods

    def reset_records(self):
        self.results = []
        self.sets = []
        self.method_names = {}

    """ Generate nodes at random from the set of locally minimal seeds according
    to a vertex-based feature. By default, that feature is neighborhood conductance,
    following

        Vertex neighborhoods, low conductance cuts, and good seeds for local community methods
        DF Gleich and C Seshahdri, KDD 2012.

    But, by providing any vector of data for each node in the graph,
    this code can be extended. Note that the conductance of each vertex
    neighborhood in the graph is calculated the first time and cached,
    which could cause a long initialization.

    @param ratio The number of nodes to pick (if > 1.0) or the fraction
    of total nodes to pick (if <= 1.0) (no default)

    @param mindegree the minimum degree of a node to pick (default 5)

    @param feature The vector of data to use to produce the locally
    minimial seeds(default None, which means to use neighborhood conductance)

    @param strict (default True) if we use a strict minimum in the definition
    of a local minimum.
    """
    def random_localmin_nodes(self, ratio, mindegree=5, feature=None, strict=True):
        if feature is None:
            # then we use conductance
            if self.neighborhood_cond is None:
                self.neighborhood_cond = triangleclusters(self.graph)[0]
            feature = self.neighborhood_cond
        # find the set of minimal verts
        minverts = self.graph.local_extrema(feature, strict)[0]
        # filter by mindegree
        minverts = [v for v in minverts if self.graph.d[v] >= mindegree]

        if 0 < ratio <= 1:
            n_nodes = min(np.ceil(ratio*len(minverts)),len(minverts))
            n_nodes = int(n_nodes)
        elif ratio > 1.0: # this is a number of nodes
            n_nodes = min(int(ratio), len(minverts))
        else:
            raise(ValueError("the ratio parameter must be positive"))

        return np.random.choice(minverts, size=n_nodes, replace=False)

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
        if result["input_set_type"]=="node":
            return [result["input_set_params"]]
        elif result["input_set_type"]=="neighborhood":
            R = self.graph.neighbors(result["input_set_params"])
            R.append(result.input_set_params)
            return R
        elif result["input_set_type"]=="set":
            return self.sets[result["input_set_params"]].copy()
        else:
            raise(ValueError("the input_set_type is unrecognized"))

    def output_set(self, j):
        result = self.results[j] # todo check for validity
        func = result["methodfunc"]
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

    def _run_samples(self, target, list_of_sets, method, timeout, nprocs):
        # TODO, figure out how we can avoid doing this everytime...
        svardata = _ncp_worker_setup(self)
        ntotalruns = sum(len(run) for run in list_of_sets)
        with mp.Pool(processes=nprocs, initializer=_ncp_worker_init, initargs=(svardata,)) as pool:
            rvals = pool.starmap(target,
                [ (workid, setids, method, timeout) for (workid, setids) in enumerate(list_of_sets) ])
            nruns = sum(len(rval) for rval in rvals)
            assert nruns <= ntotalruns, "expected up to %i ntotalruns but got %i runs"%(ntotalruns, nruns)
            for rval in rvals:
                for r in rval:
                    # make sure that we replace with our actual methods
                    # at the moment, this should always be true.
                    # this is here in case the assert fails so we can debug
                    # https://stackoverflow.com/questions/32786078/how-to-compare-wrapped-functions-with-functools-partial
                    assert r["methodfunc"]==method or _partial_functions_equal(r["methodfunc"], method)
                    r["methodfunc"] = method
                self.results.extend(rval)
            pool.close()

    def add_random_node_samples(self, ratio=0.3, timeout=1000, nthreads=4, method=None, methodname=None):
        method = self._check_method(method, methodname)
        nodes = self.random_nodes(ratio)

        list_of_sets = [ [ [j] for j in cursplit] for cursplit in np.array_split(nodes, nthreads) ] # make a list of sets
        self._run_samples(_ncp_node_worker, list_of_sets, method, timeout, nthreads)

    def add_random_neighborhood_samples(self, ratio=0.3, timeout=1000, nthreads=4, method=None, methodname=None):
        method = self._check_method(method, methodname)
        nodes = self.random_nodes(ratio)

        list_of_sets = [ [ [j] for j in cursplit] for cursplit in np.array_split(nodes, nthreads) ] # make a list of sets
        self._run_samples(_ncp_neighborhood_worker, list_of_sets, method, timeout, nthreads)

    """ Add expansions of all the locally minimal seeds.

    This routine follows ideas from:

        Vertex neighborhoods, low conductance cuts, and good seeds for local community methods
        DF Gleich and C Seshahdri, KDD 2012.

    which grow sets of out locally minimal seed vertices. These are
    vertices whose neighborhood conductance (in that paper) were better than all
    of the other neighbors. By default, this routine will use the same
    type of expansion.


    @param feature (default None, which means to use neighborhood conductance)
    @param strict, ratio, mindegree (see random_localmin_nodes)
    @param neighborhods this determinds if we seed based on the vertex (False)
    or the vertex neighborhood (True) (default True, following Gleich & Seshadhri)
    @param timeout, nthreads, method, methodname (see add_random_node samples)
    """
    def add_localmin_samples(self,
            feature=None, strict=True, ratio=1.0, mindegree=5, # localmin parameters
            neighborhoods=True, # use neighborhood expansion
            timeout=1000, nthreads=4, method=None, methodname=None):

        method = self._check_method(method, methodname)
        nodes = self.random_localmin_nodes(ratio,
            feature=feature, strict=strict, mindegree=mindegree)

        list_of_sets = [ [ [j] for j in cursplit] for cursplit in np.array_split(nodes, nthreads) ] # make a list of sets
        if neighborhoods:
            self._run_samples(_ncp_neighborhood_worker, list_of_sets, method, timeout, nthreads)
        else:
            self._run_samples(_ncp_node_worker, list_of_sets, method, timeout, nthreads)

    def add_set_samples(self, sets, nthreads=4, method=None, methodname=None, timeout=1000):
        method = self._check_method(method, methodname)
        threads = []
        startset = len(self.sets)
        self.sets.extend(sets)
        endset = len(self.sets)

        setnos = np.array_split(range(startset,endset), nthreads) # set numbers
        self._run_samples(_ncp_set_worker, setnos, method, timeout, nthreads)

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
        # convert to human readable names
        df["method"] = df["methodfunc"].map(self.method_names)

        return df

    def approxPageRank(self,
                       gamma: float = 0.01/0.99,
                       rholist: List[float] = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4],
                       ratio: float = 0.3,
                       nthreads: int = 4,
                       timeout: float = 1000):
        alpha = 1.0-1.0/(1.0+gamma)
        if self.graph._weighted:
            funcs = {functools.partial(spectral_clustering,alpha=alpha,rho=rho,method="acl_weighted"):'acl_weighted;rho=%.0e'%(rho)
                        for rho in rholist}
        else:
            funcs = {functools.partial(spectral_clustering,alpha=alpha,rho=rho,method="acl"):'acl;rho=%.0e'%(rho)
                        for rho in rholist}
        for func in funcs.keys():
            self.add_random_node_samples(method=func,methodname=funcs[func],ratio=ratio,nthreads=nthreads,timeout=timeout/len(funcs))

    def l1reg(self,
              gamma: float = 0.01/0.99,
              rholist: List[float] = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4],
              ratio: float = 0.3,
              nthreads: int = 4,
              timeout: float = 1000):
        alpha = 1.0-1.0/(1.0+gamma)
        funcs = {functools.partial(spectral_clustering, alpha=alpha,rho=rho,method="l1reg"):'l1reg;rho=%.0e'%(rho)
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
        func = functools.partial(flow_clustering,w=w, U=U, h=h,method="crd")
        self.add_random_neighborhood_samples(method=func,methodname="crd",
                ratio=ratio,nthreads=nthreads,timeout=timeout/2)
        self.add_random_node_samples(method=func,methodname="crd",
                ratio=ratio,nthreads=nthreads,timeout=timeout/2)
        return self

    def mqi(self,
            ratio: float = 0.3,
            nthreads: int = 4,
            timeout: float = 1000):
        func = functools.partial(flow_clustering,method="mqi")
        self.add_random_neighborhood_samples(ratio=ratio,nthreads=nthreads,timeout=timeout,
                method=func,methodname="mqi")
