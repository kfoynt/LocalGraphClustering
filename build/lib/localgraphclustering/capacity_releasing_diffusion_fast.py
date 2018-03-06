from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.capacity_releasing_diffusion_cpp import capacity_releasing_diffusion_cpp

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)

class Capacity_Releasing_Diffusion_fast(GraphBase[Input, Output]):

   def __init__(self) -> None:
        """
        Initialize the Capacity_Releasing_Diffusion class.
        """

        super().__init__()


   def produce(self,
              inputs: Sequence[Input], 
              ref_nodes: Sequence[int],
              U: int = 3,
              h: int = 10,
              w: int = 2,
              iterations: int = 20) -> Sequence[Output]:

    """Description
       -----------
       
       Algorithm Capacity Releasing Diffusion for local graph clustering. This algorithm uses 
       a flow based method to push excess flow out of nodes. The algorithm is in worst-case 
       faster and stays more local than classical spectral diffusion processes.
       For more details please refere to: D. Wang, K. Fountoulakis, M. Henzinger, M. Mahoney 
       and S. Rao. Capacity Releasing Diffusion for Speed and Locality. ICML 2017.
       arXiv link: https://arxiv.org/abs/1706.05826
       
       Standard call
       -------------
       
       cut = excess_unit_flow(ref_node,A,U=3,h=10,w=2,iterations=20)
       
       Data input (mandatory)
       -----------------
       
       ref_node:  integer
                  The reference node, i.e., node of interest around which
                  we are looking for a target cluster.
                  
       A:         float, double
                  Compressed Sparse Row (CSR) symmetric matrix
                  The adjacency matrix that stores the connectivity of the graph.
                  For this algorithm the graph must be undirected and unweighted,
                  which means that matrix A must be symmetric and its elements 
                  are equal to either zero or one.
       
       Algorithm parameters (optional)
       -----------------
      
       U: integer
          default == 3
          The maximum flow that can be send out of a node for the push/relabel algorithm.
          
       h: integer
          defaul == 10
          The maximum flow that an edge can handle.
          
       w: integer
          default == 2
          Multiplicative factor for increasing the capacity of the nodes at each iteration.
          
       iterations: integer
                   default = 20
                   Maximum number of iterations of Capacity Releasing Diffusion Algorithm.
          
       For details of these parameters please refer to: D. Wang, K. Fountoulakis, M. Henzinger, 
       M. Mahoney and S. Rao. Capacity Releasing Diffusion for Speed and Locality. ICML 2017.
       arXiv link: https://arxiv.org/abs/1706.05826
       
       Output
       ------
       
       cut:  list
             A list of nodes that correspond to the cluster with the best 
             conductance that was found by the Capacity Releasing Diffusion Algorithm.
       
       Printing statements (warnings)
       ------------------------------
       
       Too much excess: Meanss that push/relabel cannot push the excess flow out of the nodes.
                        This might indicate that a cluster has been found. In this case the best
                        cluster in terms of conductance is returned.
                       
       Too much flow:   Means that the algorithm has touched about a third of the whole given graph.
                        The algorithm is terminated in this case and the best cluster in terms of 
                        conductance is returned. 
    """
    
    output = [[] for i in range(len(inputs))]

    counter = 0

    for i in range(len(inputs)):

        n = inputs[i].adjacency_matrix.shape[0]

        cut = capacity_releasing_diffusion_cpp(n,np.uint32(inputs[i].adjacency_matrix.indptr),
            np.uint32(inputs[i].adjacency_matrix.indices),np.float64(inputs[i].adjacency_matrix.data),
            U,h,w,iterations,[ref_nodes[i]],inputs[i].lib)
            
        output[counter] = cut

        counter += 1

    return output