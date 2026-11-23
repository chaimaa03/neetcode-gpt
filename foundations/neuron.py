import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        Y_pred=np.dot(x,w)+b
        
        
        if activation=="sigmoid":
            Y_pred=1 / (1 + np.exp(-Y_pred))
        else:
            Y_pred=np.maximum(0,Y_pred)

        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        return round(Y_pred, 5)
        
