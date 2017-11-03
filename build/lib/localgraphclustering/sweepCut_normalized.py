from typing import *
import numpy as np
from .interface.graph import GraphBase
from .interface.types.graph import Graph
from localgraphclustering.sweep_normalized import sweep_normalized

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output',bound=np.ndarray)


class SweepCut_normalized(GraphBase[Input, Output]):

    def __init__(self) -> None:
        """
        Initialize the sweepCut_general class.
        """

        super().__init__()

    def produce(self, 
                inputs: Sequence[Input], 
                timeout: float = 100, 
                iterations: int = 1000,
                p: np.ndarray = None) -> Sequence[Output]:
        """
        It implements a sweep cut rounding procedure for local graph clustering. 
        Each component of the input vector p is divided with the corresponding degree of the node. 

        Parameters
        ----------

        inputs: Sequence[Graph]
        
        timeout: not used
        
        iterations: not used

        p: np.ndarray
           A vector that is used to perform rounding.
            
        Returns
        -------
        
        For each graph in inputs it returns in a list of length 3 the following:
        
        output 0: list
            Stores indices of the best clusters found by the last called rounding procedure.
           
        output 1: float
            Stores the value of the best conductance found by the last called rounding procedure.
                         
        output 2: list of objects
            A two dimensional list of objects. For example,
            sweep_profile[0] contains a numpy array with all conductances for all
            clusters that were calculated by the last called rounding procedure.
            sweep_profile[1] is a multidimensional list that contains the indices
            of all clusters that were calculated by the rounding procedure. For example,
            sweep_profile[1,5] is a list that contains the indices of the 5th cluster
            that was calculated by the rounding procedure. 
            The set of indices in sweep_profile[1][5] also correspond 
            to conductance in sweep_profile[0][5].
        """ 
        
        return [sweep_normalized(p,input) for input in inputs]

