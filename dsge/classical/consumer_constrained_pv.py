import numpy as np
from scipy import optimize

from dsge._base import _BaseDSGE


class ConsumerConstrainedPV(_BaseDSGE):
    """
    References
    ----------
    .. [1] https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4171
    """

    def __init__(self, W=1.0, R=1.0, beta=0.1, T=10):
        """

        Parameters
        ----------
        W: float
            Wage
        R: float
            Interest rate
        beta: float
            Discount factor
        T: int
            Number of periods
        """
        super().__init__(beta, T)
        self.W = W
        self.R = R
        self.solution_method = 'ls'

    def render(self, mode="human"):
        pass

    def solve(self):
        solution = optimize.least_squares(self.euler, x0=np.ones(self.T), bounds=(0, np.inf))
        self.c = solution.x

    def utility(self, c):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return np.log(c + 1e-9)

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
        euler = self.utility_grad(c[:-1]) - self.Beta * self.R * self.utility_grad(c[1:])
        Rt = self.R ** (1 - (np.arange(self.T) + 1))
        budget = np.array([np.dot(c, Rt) - self.W])
        return np.concatenate((euler, budget))


if __name__ == "__main__":
    model = ConsumerConstrainedPV()
    model.solve()
    print()
