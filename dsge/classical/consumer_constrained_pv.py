import numpy as np
from scipy import optimize


class ConsumerConstrainedPV:
    """
    References
    ----------
    .. [1] https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4171
    """
    def __init__(self, W=1.0, R=1.0, Beta=0.1, T=10):
        self.W = W
        self.R = R
        self.Beta = Beta
        self.T = T
        self.solution_method = 'ls'

    def solve(self):
        """
        Solves the constrained consumer problem
        """
        solution = optimize.least_squares(self.euler, x0=np.ones(self.T), bounds=(0, np.inf))
        self.c=solution.x

    def utility(self, c):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return np.log(c+1e-9)

    def utility_grad(self, c):
        """
        Gradient of utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return np.log(c)

    def euler(self, c):
        """
        Euler equation for consumption
        """
        euler=self.utility_grad(c[:-1]) - self.Beta*self.R*self.utility_grad(c[1:])
        Rt=self.R**(1-(np.arange(self.T)+1))
        budget=np.array([np.dot(c,Rt)-self.W])
        return np.concatenate((euler, budget))

