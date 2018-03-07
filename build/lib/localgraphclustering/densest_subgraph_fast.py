from typing import *
from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.acl_list import acl_list
from localgraphclustering.densest_subgraph_cpp import densest_subgraph_cpp

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class densest_subgraph_fast(GraphBase[Input, Output]):
    """
    Example of a primitive wrapping the ACL algorithm.
    """

    def __init__(self) -> None:
        """
        Initialize the approximate_PageRank class.
        """

        super().__init__()

    def produce(self,
                inputs: Sequence[Input]
                ) -> Sequence[Output]:
        """
        Finding a maximum density subgraph.
        For details please refer to: A.V.Goldberg. Finding a maximum density subgraph
        link: http://digitalassets.lib.berkeley.edu/techreports/ucb/text/CSD-84-171.pdf

        Parameters
        ----------

        inputs: Sequence[Graph]
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        An np.ndarray (1D embedding) of the nodes for each graph with low conductance.

        """ 
            
        output = [[] for i in range(len(inputs))]

        counter = 0

        for input in inputs:

            n = input.adjacency_matrix.shape[0]

            (density,actual_xids) = densest_subgraph_cpp(n,np.int64(input.adjacency_matrix.indptr),
                np.int64(input.adjacency_matrix.indices),np.float64(input.adjacency_matrix.data/2),input.lib)
            
            output[counter] = [density,actual_xids]

            counter += 1

        return output
