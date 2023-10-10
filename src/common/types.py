from typing import Union
import jax
import numpy as np
from numpy.typing import NDArray

AnyArray = Union[NDArray, jax.Array]
AnyFloatingArray = Union[NDArray[np.floating], jax.Array]
