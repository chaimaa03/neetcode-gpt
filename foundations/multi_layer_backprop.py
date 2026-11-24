import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)
        z1=W1@x+b1
        a=np.maximum(0,z1)
        z2=W2@a+b2
        # Loss: MSE = mean((predictions - y_true)^2)
        loss=np.mean((z2-y_true)**2)
        
        dZ2=2*(z2-y_true)
        dW2=np.outer(dZ2,a)
        db2=dZ2
        da=np.dot(W2.T,dZ2)
        dZ1=da*(z1>0).astype(float)
        dW1=np.outer(dZ1,x)
        db1=dZ1
        

        return {
           'loss':  np.round(loss,4).tolist(),
           'dW1':   np.round(dW1,4).tolist(),
           'db1':   np.round(db1,4).tolist(),
           'dW2':   np.round(dW2,4).tolist(),
           'db2':   np.round(db2,4).tolist()
        }
