import numpy as np
from typing import List, Type, Union


class Discrete:
    """
        Each point is mapped to an integer from [0 ,n−1]
        Discrete(10) A space containing 10 items mapped to integers in [0,9] sample will return integers such as 0, 3, and 9.
        """

    def __init__(self, n: int, start: int = 0, dtype: Type = np.uint64):
        self.n = n
        self.start = start
        self.dtype = dtype

    def sample(self):
        return int(self.start + np.random.randint(self.n))


class MultiDiscrete:
    """This represents the cartesian product of arbitrary :class:`Discrete` spaces.

    It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space.

    Note:
        Some environment wrappers assume a value of 0 always represents the NOOP action.

    e.g. Nintendo Game Controller - Can be conceptualized as 3 discrete action spaces:

    1. Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
    2. Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    3. Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    It can be initialized as ``MultiDiscrete([ 5, 2, 2 ])``
    """

    def __init__(self, nvec: Union[np.ndarray, List[int]], dtype: Type = np.uint64):
        self.nvec = np.array(nvec)
        self.dtype = dtype

    def sample(self):
        return (np.random.random(size=self.nvec.shape) * self.nvec).astype(self.dtype)


class Box:
    """
    Used for multidimensional continuous spaces with bounds
    Box(np.array((-1.0, -2.0)), np.array((1.0, 2.0)))  A 2D continous state spaceI
    First dimension has values in range [−1.0,1.0)
    Second dimension has values in range [−2.0,2.0) sample will return a vector such as [−.55,2.] and [.768,−1.55]
    """

    def __init__(self, low: Union[List, np.ndarray],
                 high: Union[List, np.ndarray], dtype: Type = np.float32):
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype

        assert len(low) == len(high)
        self.shape = [len(low)]

    def sample(self):
        x = (self.high - self.low) * np.random.random(size=self.shape[0]) + self.low
        return x


class Tuple:
    pass
