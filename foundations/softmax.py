import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        p= z - np.max(z)
        S=np.exp(p)/np.sum(np.exp(p))
        return np.round(S, 4)
        
