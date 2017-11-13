from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph

from localgraphclustering.ncp_MQI_algo import ncp_MQI_algo

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class Ncp_MQI(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the Ncp class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input], 
                ratio: float = 0.3,
                timeout_ncp = 1000,
                ) -> Sequence[Output]:
        """
        Network Community Profile based on neighborhoods and MQI for all connected components of the graph.

        Parameters
        ----------  

        ratio: float
            Ratio of nodes to be used for computation of NCP.
            It should be between 0 and 1.

        Parameters (optional)
        ---------------------

        timeout_ncp: float
            default = 1000
            Maximum time in seconds for NCP calculation.

        Returns
        -------
        
        For each graph in inputs it returns the following:

    conductance_vs_vol_R: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. The clusters are computed using neighbors of randomly 
        selected nodes. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.

    isoperimetry_vs_size_R: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are isoperimetry. The clusters are computed using neighbors of randomly 
        selected nodes. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.
        
    conductance_vs_node_R: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. The clusters are computed using neighbors of randomly 
        selected nodes. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.

    isoperimetry_vs_node_R: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are isoperimetry. The clusters are computed using neighbors of randomly 
        selected nodes. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.
        
    conductance_vs_vol_MQI: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. The clusters are computed using neighbors of randomly 
        selected nodes on which we apply MQI. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.

    isoperimetry_vs_size_MQI: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are isoperimetry. The clusters are computed using neighbors of randomly 
        selected nodes on which we apply MQI. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.
        
    conductance_vs_node_MQI: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are volumes of clusters and 
        the values are conductance. The clusters are computed using neighbors of randomly 
        selected nodes on which we apply MQI. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.

    isoperimetry_vs_node_MQI: a list of dictionaries
        The length of the list is the number of connected components of the given graph.
        Each element of the list is a dictionary where keys are sizes of clusters and 
        the values are isoperimetry. The clusters are computed using neighbors of randomly 
        selected nodes on which we apply MQI. If ratio=1, then all nodes are selected. 
        It can be used to plot the conductance vs volume NCP.
        """       
        
        return [ncp_MQI_algo(inputs[i], ratio=ratio, timeout_ncp=timeout_ncp) for i in range(len(inputs))]


