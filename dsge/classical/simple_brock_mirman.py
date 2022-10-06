import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dsge._base import _BaseDSGE


class SimpleBrockMirman(_BaseDSGE):
    """
    References
    ----------
    .. [1] http://www.econ2.jhu.edu/people/ccarroll/public/LectureNotes/DSGEModels/BrockMirman/
    """

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=10, G=0.02):
        """

        Parameters
        ----------
        alpha: float
            Capital share of output
        beta: float
            Discount factor
        K_0: float
            Initial capital stock
        A_0: float
            Initial technology level
        T: int
            Number of periods
        G: float
            Growth rate of technology
        """
        super().__init__(beta, T)
        self.alpha = alpha
        self.K_0 = K_0
        self.A_0 = A_0
        self.c = np.zeros(self.T)
        self.k = np.zeros(self.T)
        self.y = np.zeros(self.T)
        self.l = np.ones(self.T)
        self.k[0] = K_0
        self.A = self.A_0 * np.cumprod(np.ones(self.T) * (1 + G)) / (1 + G)
        self.delta = 1  # assumption of Brock-Mirman

    def render(self):
        df = self._history()
        df = pd.melt(df, id_vars=['time'], value_vars=['capital', 'consumption', 'technology', 'labour'])
        plt.figure()
        gfg = sns.lineplot(data=df, x='time', y='value', hue='variable')
        gfg.set_ylim(bottom=0)

    def _history(self):
        df = pd.DataFrame(
            {'time': self.t, 'consumption': self.c, 'capital': self.k, 'technology': self.A, 'labour': self.l})
        return df

    def capital_accumulation(self, K, Y, C):
        return (1 - self.delta) * K + Y - C

    def production(self, A, K, **kwargs):
        return A * K ** self.alpha

    def utility(self, c, **args):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return np.log(c + 1e-9)

    def total_utility(self, c, **args):
        pv_utility = self.utility(c) * (self.beta ** self.t)
        return np.sum(pv_utility)

    def solve(self):
        """
        Solves the constrained consumer problem
        """
        self.solve_closed_form()

    def solve_closed_form(self):
        kappa = 1 - self.alpha * self.beta
        for t in range(1, self.T):
            self.k[t] = self.alpha * self.beta * self.A[t - 1] * self.k[t - 1] ** self.alpha
        self.y[:] = self.A * self.k ** self.alpha
        self.c[:] = kappa * self.y


if __name__ == "__main__":
    model = SimpleBrockMirman()
    model.solve()
