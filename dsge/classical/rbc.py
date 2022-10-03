import numpy as np
from scipy import optimize

from dsge._base import _BaseDSGE


class RBC(_BaseDSGE):
    """
    Real Business Cycle Model

    References
    ----------
    .. [1] https://personal.lse.ac.uk/vernazza/_private/RBC%20Models.pdf
    """

    def __init__(self, alpha=0.5, beta=0.5, delta=0.5, K_0=1, A_0=1, T=10, G=0.02,b=0.5,gamma=0.5, solution='closed_form'):
        """

        Parameters
        ----------
        alpha: float
            Capital share of output
        beta: float
            Discount factor
        delta: float
            Depreciation rate
        K_0: float
            Initial capital stock
        A_0: float
            Initial technology level
        T: int
            Number of periods
        G: float
            Growth rate of technology
        b: float
            Leisure preference
        gamma: float
            Elasticity of substitution between consumption and leisure
        """
        super().__init__(beta, T)
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.solution = solution
        self.Y = np.zeros(self.T)
        self.C = np.zeros(self.T)
        self.K = np.zeros(self.T)
        self.K_0 = K_0
        self.A_0 = A_0
        self.K[0] = K_0
        self.A = self.A_0*np.cumprod(np.ones(self.T) *(1+ G))/(1+G)
        self.b=b

    def production(self, A, K, N=1):
        return K ** (1-self.alpha) * (A * N) ** self.alpha

    def capital_accumulation(self, K, Y, C):
        return (1 - self.delta) * K + Y - C

    def utility(self, c, l):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        l : float
            Leisure
        """
        return np.log(c + 1e-9)+self.b*np.log(1-l+1e-9)

    def solve(self):
        """
        Solves the constrained consumer problem
        """
        if self.solution == 'lsq':
            solution = optimize.least_squares(self.euler, x0=np.ones(self.T), bounds=(0, np.inf))
        elif self.solution == 'taylor':
            raise NotImplementedError


if __name__ == "__main__":
    model = RBC()
    model.solve()
    print()
