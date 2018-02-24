import abc
from typing import *

from .types.graph import Graph

__all__ = ('GraphBase',)

Input = TypeVar('Input', bound=Graph)
Output = TypeVar('Output')


class GraphBase(Generic[Input, Output]):
    """
    A base class for graph methods which take Graph objects as input.
    Graph is an interface which TA1 teams should implement for graph data.
    """
    
    def __init__(self) -> None:
        """
        All primitives should specify all the hyper-parameters that can be set at the class
        level in their ``__init__`` as explicit typed keyword-only arguments
        (no ``*args`` or ``**kwargs``).

        Hyper-parameters are those primitive's parameters which are not changing during
        a life-time of a primitive. Parameters which do are set using the ``set_params`` method.
        """

    @abc.abstractmethod
    def produce(self, *, inputs: Sequence[Input], timeout: float = None, iterations: int = None) -> Sequence[Output]:
        """
        This function should be implemented by subclasses.

        Parameters
        ----------
        inputs : Sequence[Input]
            The inputs of shape [num_inputs, ...].
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        Sequence[Output]
            The outputs of shape [num_inputs, ...].
        """