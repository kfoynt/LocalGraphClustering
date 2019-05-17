from typing import *
import numpy as np
import pandas as pd

import time
import threading
import math
import warnings
import copy
import multiprocessing as mp
import functools
import pickle

from .spectral_clustering import spectral_clustering
from .flow_clustering import flow_clustering
from .GraphLocal import GraphLocal
from .triangleclusters import triangleclusters
from .cpp import *

np.random.seed(seed=123)

class partialfunc(functools.partial):
    @classmethod
    def from_partial(cls, f):
        print("f.func: ", f.func, " f.args: ", f.args, " f.keywords: ", f.keywords)
        return cls(f.func, f.args, f.keywords)
    def __eq__(self, f2):
        if not (isinstance(f2, partialfunc)):
            return False
        return all([getattr(self, attr) == getattr(f2, attr) for attr in ['func', 'args', 'keywords']])
    __hash__ = functools.partial.__hash__

class SimpleLogForLongComputations:
    """ Implement a simple logger that will record messages and then
    replay them if a timer exceeds a threshold."""
    def __init__(self, mintime=120, method=""):
        if len(method) > 0:
            self._prefix = method + ": "
        else:
            self._prefix = ""
        self._log = []
        self._mintime = mintime
        self._t0 = time.time()

    def _print_message(self, t, m):
        print(self._prefix,end="")
        print("%6.1f"%(t - self._t0), end=" ")
        print(m)

    def dumplog(self):
        """ Dump the log (i.e. print it) and clear the messages. """
        if len(self._log) > 0:
            for (t, m) in self._log:
                self._print_message(t, m)
            self._log = [] # reset the log.

    def log(self, message):
        """ Log a message, which is spooled unless the mintime is exceeded. """
        t = time.time()
        if t - self._t0 > self._mintime:
            self.dumplog()
            self._print_message(t,message)
        else:
            self._log.append((t, message))

def _partial_functions_equal(func1, func2):
    assert(False) # shouldn't be called now
    if not (isinstance(func1, functools.partial) and isinstance(func2, functools.partial)):
        return False
    are_equal = all([getattr(func1, attr) == getattr(func2, attr) for attr in ['func', 'args', 'keywords']])
    if are_equal == False:
        # TODO remove this code once we are sure things are working okay
        print(func1, "!=", func2)
        for attr in ['func', 'args', 'keywords']:
            print(getattr(func1, attr), getattr(func2, attr))
    return are_equal

def does_nothing():
    return

""" This is helpful for some of the NCP studies to return the set we are given. """
def _evaluate_set(G,N):
    if 0 < len(N) < G._num_vertices:
        return N, None
    else:
        return [], None

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

    if len(S) > 0 and sum(ncpdata.graph.d[S]) < ncpdata.graph.vol_G/2:
        output_stats = ncpdata.graph.set_scores(S)
        for F in ncpdata.set_funcs: # build the list of keys for set_funcs
            output_stats.update(F(ncpdata.graph, S))
        output_stats = {"output_" + str(key):value for key,value in output_stats.items() } # add output prefix
        if ncpdata.store_output_clusters:
            output_cluster = {"output_cluster": S}

        method_stats['methodfunc']  = func
        method_stats['time'] = dt
        if ncpdata.store_output_clusters:
            return [dict(**input_stats, **output_stats, **method_stats, **output_cluster)]
        else:
            return [dict(**input_stats, **output_stats, **method_stats)]
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

def _ncp_localworker_init(ncpdata):
    # no need to set the graph as we haven't cleared it from the local one
    workvars["ncpdata"] = ncpdata

def _ncp_node_worker(workid, setids,func,timeout):
    return ncp_worker(workid, "node", workvars["ncpdata"], setids, func, timeout)

def _ncp_neighborhood_worker(workid, setids,func,timeout):
    return ncp_worker(workid, "neighborhood", workvars["ncpdata"], setids, func, timeout)

def _ncp_refine_worker(workid, setids, func, timeout):
    return ncp_worder(workid, "refine", workvars["ncpdata"], setids, func, timeout)

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
        elif runtype == 'refine':
            setid = Rid
            R = ncpdata.output_set(setid)[0]
            method_stats = {'input_set_type': runtype, 'input_set_params':Rid}
            result = ncp_experiment(ncpdata, R, func, method_stats)
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
            raise(ValueError("the runtype must be 'node','refine','neighborhood', or 'set'"))

        end = time.time()
        if end - start > timeout:
            break
    return results

class NCPData:
    def __init__(self, graph, setfuncs=[], input_stats=True, do_largest_component=True, store_output_clusters=False):
        if do_largest_component:
            self.graph = graph.largest_component()
        else:
            self.graph = graph

        # We need to save this for the pickle input/output
        self.do_largest_component = do_largest_component
        self.input_stats = input_stats
        self.store_output_clusters = store_output_clusters
        self.set_funcs = setfuncs
        self.nruns = 0

        standard_fields = self.graph.set_scores([0])
        for F in self.set_funcs: # build the list of keys for set_funcs
            standard_fields.update(F(self.graph, [0]))
        if self.input_stats:
            result_fields = ["input_" + field for field in standard_fields.keys()]
        result_fields.extend(["output_" + str(field) for field in standard_fields.keys()])
        result_fields.extend(["methodfunc", "input_set_type", "input_set_params", "time"])
        if store_output_clusters:
            result_fields.extend(["output_cluster"])
        self.result_fields = result_fields
        self.neighborhood_cond = None
        self.fiedler_set = None
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
        nminverts = len(minverts)
        if nminverts == 0:
            warnings.warn("There are no localmin nodes")
            return np.array([],'int64')
        # filter by mindegree
        minverts = [v for v in minverts if self.graph.d[v] >= mindegree]
        if len(minverts) == 0:
            warnings.warn("there are %i localmin nodes and they were filtered away by mindegree=%i"%(
                nminverts, mindegree))
            return np.array([],'int64')

        assert(len(minverts) > 0)

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
            R.append(result["input_set_params"])
            return R
        elif result["input_set_type"]=="set":
            return self.sets[result["input_set_params"]].copy()
        elif result["input_set_type"]=="refine":
            return self.output_set(result["input_set_params"])[0]
        else:
            raise(ValueError("the input_set_type is unrecognized"))

    def output_set(self, j):
        assert(self.graph is not None)
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
        if nprocs == 1:
            # we special case nprocs = 1 so that we can get coverage
            # of the code in our report.
            _ncp_localworker_init(self)
            ntotalruns = sum(len(run) for run in list_of_sets)
            nruns = 0
            for (workid, setids) in enumerate(list_of_sets):
                rval = target(workid, setids, method, timeout)
                nruns += len(rval)
                assert nruns <= ntotalruns, "expected up to %i ntotalruns but got %i runs"%(ntotalruns, nruns)
                self.results.extend(rval)
        else:
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
    @param neighborhoods this determinds if we seed based on the vertex (False)
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

    def add_set_samples_without_method(self, sets):
        method = self._check_method(does_nothing, None)
        startset = len(self.sets)
        self.sets.extend(sets)

        counter = 1
        for cluster in sets:
            input_stats = self.graph.set_scores(cluster)
            for F in self.set_funcs: # build the list of keys for set_funcs
                input_stats.update(F(self.graph, cluster))
            input_stats = {"input_" + str(key):value for key,value in input_stats.items() } # add input prefix

            output_stats = self.graph.set_scores(cluster)
            for F in self.set_funcs: # build the list of keys for set_funcs
                output_stats.update(F(self.graph, cluster))
            output_stats = {"output_" + str(key):value for key,value in output_stats.items() } # add output prefix
            if self.store_output_clusters:
                output_cluster = {"output_cluster": cluster}

            method_stats = {'input_set_type': 'set', 'input_set_params':startset+counter, 'methodfunc':does_nothing, 'time':0}

            self.results.extend([dict(**input_stats, **output_stats, **method_stats)])

            counter = counter + 1

    def as_data_frame(self):
        """ Return the NCP results as a pandas dataframe """
        df = pd.DataFrame.from_records(self.results, columns=self.result_fields)
        # convert to human readable names
        # It's important that this dictionary is converted into a lookup
        # function so the pandas map function works correctly with our
        # partial functions that may hash differently but compare as equal.
        # Ideally, we'd call...
        # df["method"] = df["methodfunc"].map(self.method_names)
        df["method"] = df["methodfunc"].map(lambda x: self.method_names[x])
        # TODO, since this is a bit hacky, it's probably worth storing
        # the method name in the results itself. That's probably better at
        # this point

        return df

    def write(self, filename: str, writepython: bool = True, writecsv: bool = True):
        """ Write the NCP data to a fileself.

        writepython: True if the current class should be pickled to a fileself.
        writecsv: True if the current results should be written to a CSV file.

        Note that the python output can be used in ways that the CSV output
        cannot. For instance, we don't store the "sets" used to build
        the data in the CSV output.
        """
        # We pickle ncpdata to a file
        # temporarily remove graph, so that isn't pickled
        if writepython:
            mygraph = self.graph
            self.graph = None
            with open(filename + ".pickle", "wb") as file:
                pickle.dump(self, file)
            self.graph = mygraph
        # dump a CSV file based on the DataFrame
        if writecsv:
            self.as_data_frame().to_csv(filename + ".csv")

    @classmethod
    def from_file(cls, filename: str, g: GraphLocal):
        with open(filename, "rb") as file:
            ncp = pickle.load(file)
        if ncp.do_largest_component:
            ncp.graph = g.largest_component()
        else:
            ncp.graph = g
        return ncp

    def approxPageRank(self,
                       gamma: float = 0.01/0.99,
                       rholist: List[float] = [1.0e-5,1.0e-4],
                       localmins: bool = True,
                       localmin_ratio: float = 0.5,
                       neighborhood_ratio: float = 0.1,
                       neighborhoods: bool = True,
                       random_neighborhoods: bool = True,
                       timeout: float = 1000,
                       spectral_args: Dict = {},
                       deep: bool = False,
                       method = "acl",
                       methodname_prefix: str = "ncpapr",
                       normalize: bool = True,
                       normalized_objective: bool = True,
                       cpp: bool = True,
                       **kwargs):
        """ Compute the NCP via an approximate PageRank computation.

        This function tries to quickly approach the NCP plots from a
        longer computation by using a number of effective shortcuts.
        1. It does neighborhood sampling to generate the NCP plot
        for small sets (if neighborhoods = true)
        2. It does localmin sampling to generate the extrema of the NCP plot
        (if localmins = True). Because we use neighborhood seeds for these,
        then we increase the value of rho by a factor of 10.
        3. It runs random PPR and neighborhood seeded PPR
        samples to generate the bulk of the NCP plot
        for large-ish values of size. This will help fill in the bulk for
        medium and large cluster sizes.

        If the "deep" parameter is specified, we search for especially deep
        and large partitions. This involves additional computational expense
        in terms of smaller values of gamma and smaller values of rho.
        The defaults here are basically to search for values of at least
        10 times smaller than the default search, with gamma values about 100
        times smaller. This is a good test if you expect geometric structure
        in your data. The NCP for an information graph should be mostly
        the same with or without deep.

        """
        
        if len(methodname_prefix) > 0:
            methodname = methodname_prefix + "_" + method
            deepmethodname_prefix = "deep" + "_" + methodname_prefix
        else:
            methodname = method
            deepmethodname_prefix = "deep"

        alpha = 1.0-1.0/(1.0+gamma)

        # save the old ratio
        myratio = None
        if 'ratio' in kwargs:
            myratio = kwargs['ratio']
            del kwargs['ratio']

        deeptimeout = timeout # save the timeout in case we also run the deep params
        nruns = 1
        if random_neighborhoods:
            nruns += 1
        if localmins:
            nruns += 1

        log = SimpleLogForLongComputations(120, "approxPageRank:%s"%(methodname))

        if neighborhoods:
            self.add_random_neighborhood_samples(
                method=_evaluate_set,
                methodname="neighborhoods",
                ratio=neighborhood_ratio,timeout=timeout/10,**kwargs)
            timeout -= timeout/10
            log.log("neighborhoods")

        if localmins:
            for rho in rholist:
                self.add_localmin_samples(
                    method=partialfunc(
                        spectral_clustering,**spectral_args,alpha=alpha,rho=rho*10,
                        method=method,normalize=normalize,
                        normalized_objective=normalized_objective,cpp=cpp),
                    methodname="%s_localmin:rho=%.0e"%(methodname, rho*10),
                    neighborhoods=True,
                    ratio=localmin_ratio,
                    timeout=timeout/(nruns*len(rholist)),**kwargs)
                log.log("localmin rho=%.1e"%(rho))
            timeout -= timeout/nruns # reduce the time left...
            nruns -= 1


        for rho in rholist:
            if myratio is not None:
                kwargs['ratio'] = myratio
            self.add_random_node_samples(
                method=partialfunc(
                    spectral_clustering,**spectral_args,alpha=alpha,rho=rho,
                    method=method,normalize=normalize,
                    normalized_objective=normalized_objective,cpp=cpp),
                methodname="%s:rho=%.0e"%(methodname, rho),
                timeout=timeout/(nruns*len(rholist)), **kwargs)
            log.log("random_node rho=%.1e"%(rho))

        timeout -= timeout/nruns # reduce the time left...

        for rho in rholist:
            if myratio is not None:
                kwargs['ratio'] = myratio
            self.add_random_neighborhood_samples(
                method=partialfunc(
                    spectral_clustering,**spectral_args,alpha=alpha,rho=rho*10,
                    method=method,normalize=normalize,normalized_objective=normalized_objective,cpp=cpp),
                methodname="%s_neighborhoods:rho=%.0e"%(methodname, rho*10),
                timeout=timeout/(len(rholist)), **kwargs)
            log.log("random_neighborhood rho=%.1e"%(rho))

        if deep:
            timeout = deeptimeout
            # figure out the minimum ratio between rho steps
            rholist.sort()
            minratio = 10.0 # if the ratio is bigger than 10, we'll just use that
            for i in range(len(rholist)-1):
                if rholist[i+1]/rholist[i] < minratio:
                    minratio = rholist[i+1]/rholist[i]
            minrho = min(rholist)
            maxrho = max(rholist)
            deeprhos = [(minrho/maxrho)*rho/minratio for rho in rholist]
            deepgamma = gamma/10.0
            if 'iterations' in spectral_args: # we need more iterations
                # but not too too many
                spectral_args['iterations'] = min(1000*spectral_args['iterations'], 2000000000)
            else:
                spectral_args['iterations'] = 1000000000

            # note, ratio has already been re-added to kwargs above
            self.approxPageRank(gamma=deepgamma,rholist=deeprhos,
                localmins=localmins, localmin_ratio=localmin_ratio,
                neighborhoods=False, neighborhood_ratio=neighborhood_ratio,
                random_neighborhoods=random_neighborhoods,
                timeout = deeptimeout, spectral_args=spectral_args, deep=False,
                method=method, methodname_prefix=deepmethodname_prefix,
                normalize=normalize,normalized_objective=normalized_objective,
                **kwargs)

        return self

    def l1reg(self,
              gamma: float = 0.01/0.99,
              rholist: List[float] = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4],
              ratio: float = 0.3,
              nthreads: int = 4,
              timeout: float = 1000):
        alpha = 1.0-1.0/(1.0+gamma)
        funcs = {partialfunc(spectral_clustering, alpha=alpha,rho=rho,method="l1reg"):'l1reg;rho=%.0e'%(rho)
                    for rho in rholist}
        for func in funcs.keys():
            self.add_random_node_samples(method=func,methodname=funcs[func],ratio=ratio,nthreads=nthreads,timeout=timeout/len(funcs))
        return self

    def l1reg_rand(self,
              gamma: float = 0.01/0.99,
              rholist: List[float] = [1.0e-10,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4],
              ratio: float = 0.3,
              nthreads: int = 4,
              timeout: float = 1000):
        alpha = 1.0-1.0/(1.0+gamma)
        funcs = {partialfunc(spectral_clustering, alpha=alpha,rho=rho,method="l1reg-rand"):'l1reg;rho=%.0e'%(rho)
                    for rho in rholist}
        for func in funcs.keys():
            self.add_random_node_samples(method=func,methodname=funcs[func],ratio=ratio,nthreads=nthreads,timeout=timeout/len(funcs))
        return self

    def crd(self,
            U: int = 3,
            h: int = 10,
            w: int = 2,
            ratio: float = 0.3,
            nthreads: int = 4,
            timeout: float = 1000):
        func = partialfunc(flow_clustering,w=w, U=U, h=h,method="crd")
        self.add_random_neighborhood_samples(method=func,methodname="crd",
                ratio=ratio,nthreads=nthreads,timeout=timeout/2)
        self.add_random_node_samples(method=func,methodname="crd",
                ratio=ratio,nthreads=nthreads,timeout=timeout/2)
        return self

    def mqi(self,
            ratio: float = 0.3,
            nthreads: int = 4,
            timeout: float = 1000):
        func = partialfunc(flow_clustering,method="mqi")
        self.add_random_neighborhood_samples(ratio=ratio,nthreads=nthreads,timeout=timeout,
                method=func,methodname="mqi")
        return self

    def refine(self,
            sets,
            method=None,
            methodname=None,
            nthreads: int = 4,
            timeout: float = 1000,
            **kwargs):
        method = self._check_method(method, methodname)
        func = partialfunc(flow_clustering,delta=kwargs["delta"],method=method)
        return self.add_set_samples(methodname=methodname,method=func, nthreads=nthreads, sets=sets, timeout=timeout)

    def _fiedler_set(self):
        if self.fiedler_set is None:
            self.fiedler_set = spectral_clustering(self.graph, None, method="fiedler")[0]
        return self.fiedler_set

    def add_fiedler(self):
        S = self._fiedler_set()
        # note that we use functools partial here to create a new function
        # that we name "fiedler" even though the code is just evaluate_set
        return self.add_set_samples(methodname="fiedler",
            method=partialfunc(_evaluate_set), nthreads=1, sets=[S])

    def add_fiedler_mqi(self):
        S = self._fiedler_set()
        return self.add_set_samples(methodname="fiedler-mqi",
            method=partialfunc(flow_clustering,method="mqi"), nthreads=1, sets=[S])

    def add_neighborhoods(self, **kwargs):
        return self.add_random_neighborhood_samples(
            method=_evaluate_set,methodname="neighborhoods",**kwargs)
