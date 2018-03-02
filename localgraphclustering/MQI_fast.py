from typing import *
from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.acl_list import acl_list
from localgraphclustering.MQI_cpp import MQI_cpp

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class MQI_fast(GraphBase[Input, Output]):
    """
    Example of a primitive wrapping the ACL algorithm.
    """

    def __init__(self) -> None:
        """
        Initialize the approximate_PageRank class.
        """

        super().__init__()

    def produce(self,
                inputs: Sequence[Input], 
                ref_node: Sequence[Sequence[int]]
                ) -> Sequence[Output]:
        """
        Max Flow Quotient Cut Improvement (MQI) for improving either the expansion or the 
        conductance of cuts of weighted or unweighted graphs.
        For details please refer to: Lang and Rao (2004). Max Flow Quotient Cut Improvement (MQI)
        link: https://link.springer.com/chapter/10.1007/978-3-540-25960-2_25

        Parameters
        ----------

        inputs: Sequence[Graph]

        ref_node: Sequence[Sequence[int]]
            The reference node, i.e., node of interest around which
            we are looking for a target cluster.
            
        Returns
        -------
        
        For each graph in inputs it returns the following:
        
        An np.ndarray (1D embedding) of the nodes for each graph with low conductance.

        """ 
            
        output = [[] for i in range(len(inputs))]

        counter = 0

        for input in inputs:

            n = input.adjacency_matrix.shape[0]

            R = list(set(ref_node[counter]))

            nR = len(R)

            (actual_length,actual_xids) = MQI_cpp(n,np.uint32(input.adjacency_matrix.indptr),
                np.uint32(input.adjacency_matrix.indices),nR,R,input.lib)
            
            output[counter] = [actual_xids]

            counter += 1

        return output
