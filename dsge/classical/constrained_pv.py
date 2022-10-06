import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize

from dsge._base import _BaseDSGE


class ConstrainedPV(_BaseDSGE):
    """
    References
    ----------
    .. [1] https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4171
    """

    def __init__(self, W=1.0, R=1.0, beta=0.1, T=10, eps=1e-3):
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
        self.c = np.zeros(self.T)
        self.solution_method = 'ls'
        self.eps = eps

    def _history(self):
        df = pd.DataFrame({'time': self.t, 'consumption': self.c})
        return df

    def render(self):
        df = self._history()
        df = pd.melt(df, id_vars=['time'], value_vars=['consumption'])
        plt.figure()
        gfg = sns.lineplot(data=df, x='time', y='value', hue='variable')
        gfg.set_ylim(bottom=0)

    def solve(self):
        solution = optimize.least_squares(self.euler, x0=np.zeros(self.T), bounds=(0, np.inf), max_nfev=10000)
        self.c = solution.x

    def utility(self, c):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return np.log(c + self.eps)

    def total_utility(self, c):
        pv_utility = self.utility(c) * (self.beta ** self.t)
        return np.sum(pv_utility)

    def utility_grad(self, c):
        """
        Gradient of utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return 1 / (c + self.eps)

    def budget(self, c):
        """
        Budget constraint
        """
        Rt = self.R ** (1 - (np.arange(self.T) + 1))
        return np.array([np.dot(c, Rt) - self.W])

    def euler(self, c):
        """
        Euler equation for consumption
        """
        euler = self.utility_grad(c[:-1]) - self.beta * self.R * self.utility_grad(c[1:])
        return np.concatenate((euler, self.budget(c)))


if __name__ == "__main__":
    model = ConstrainedPV()
    model.solve()
    print()
