class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: 
        x=init 
        def f(x):
            return x**2
        # Derivative:         
        def f_deriv(x):
            return 2*x
        # Update rule: 
        
        for i in range(iterations):
            x= x - learning_rate * f_deriv(x)
        # Round final answer to 5 decimal places
        return round(x,5)

