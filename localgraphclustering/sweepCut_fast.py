from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.sweepcut_cpp import sweepcut_cpp

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class SweepCut_fast(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the sweepCut_general class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input], 
                timeout: float = 100, 
                iterations: int = 1000,
                p: np.ndarray = None,
                do_sort: bool = True) -> Sequence[Output]:
        """
        It implements a sweep cut rounding procedure for local graph clustering. 

        Parameters
        ----------

        inputs: Sequence[Graph]
        
        timeout: not used
        
        iterations: not used

        p: np.ndarray
           A vector that is used to perform rounding.
           
        do_sort: binary 
            default = 1 
            If do_sort is equal to 1 then vector p is sorted in descending order first.
            If do_sort is equal to 0 then vector p is not sorted.
            
        Returns
        -------
        
        For each graph in inputs it returns in a list of length 2 the following:
        
        output 0: list
            Stores indices of the best clusters found by the last called rounding procedure.
           
        output 1: float
            Stores the value of the best conductance found by the last called rounding procedure.
        """ 
        
        output = [[] for i in range(len(inputs))]
        
        counter = 0
        
        for input in inputs:
        
            n = input.adjacency_matrix.shape[0]

            nnz_idx = p.nonzero()[0]
            nnz_ct = nnz_idx.shape[0]

            if nnz_ct == 0:
                output[counter] = [[],0]
                counter += 1
                continue

            sc_p = np.zeros(nnz_ct)
            for i in range(nnz_ct):
                sc_p[i] = p[nnz_idx[i]]   

            (actual_length,bestclus,best_conductance) = sweepcut_cpp(n, 
                np.uint32(input.adjacency_matrix.indptr), np.uint32(input.adjacency_matrix.indices), 
                input.adjacency_matrix.data, nnz_idx, nnz_ct, sc_p, 1 - do_sort,input.lib)

            best_cluster = bestclus.tolist()
            
            output[counter] = [best_cluster,best_conductance]
            
            counter = counter + 1
            
        return output

