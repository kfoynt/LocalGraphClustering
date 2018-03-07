from typing import *
from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.SimpleLocal_cpp import SimpleLocal_cpp

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class SimpleLocal_fast(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the approximate_PageRank class.
        """

        super().__init__()

    def produce(self,
                inputs: Sequence[Input], 
                ref_node: Sequence[Sequence[int]],
                delta: float = 0.3
                ) -> Sequence[Output]:
        """
        A Simple and Strongly-Local Flow-Based Method for Cut Improvement.
        For details please refer to: Veldt, Gleich and Mahoney (2016).
        link: https://arxiv.org/abs/1605.08490

        Parameters
        ----------

        inputs: Sequence[Graph]

        ref_node: Sequence[Sequence[int]]
            The reference node, i.e., node of interest around which
            we are looking for a target cluster.

        delta: float
            locality parameter
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        An np.ndarray (1D embedding) of the nodes for each graph with low conductance.

        """ 
            
        output = [[] for i in range(len(inputs))]

        counter = 0

        for input in inputs:

            n = input.adjacency_matrix.shape[0]

            (actual_length,actual_xids) = SimpleLocal_cpp(n,np.int64(input.adjacency_matrix.indptr),
                np.int64(input.adjacency_matrix.indices),len(ref_node[counter]),ref_node[counter],delta,input.lib)
            
            output[counter] = [actual_xids]

            counter += 1

        return output
