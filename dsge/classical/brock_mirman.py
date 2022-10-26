import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dsge._base import _BaseDSGE


class BrockMirman(_BaseDSGE):
    """
    References
    ----------
    .. [1] https://personal.lse.ac.uk/vernazza/_private/RBC%20Models.pdf
    """

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=10, G=0.02, b=0.5, solver='closed form'):
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
        super().__init__(beta, T, solver=solver)
        self.alpha = alpha
        self.K_0 = K_0
        self.A_0 = A_0
        self.c = np.zeros(self.T)
        self.k = np.zeros(self.T)
        self.y = np.zeros(self.T)
        self.l = np.ones(self.T)
        self.r = np.zeros(self.T)
        self.w = np.zeros(self.T)
        self.k[0] = K_0
        self.A = self.A_0 * np.cumprod(np.ones(self.T) * (1 + G)) / (1 + G)
        self.delta = 1  # assumption of Brock-Mirman
        self.b = b

    def render(self):
        df = self.history
        df = pd.melt(df, id_vars=['time'], value_vars=['capital', 'consumption', 'technology', 'labour'])
        plt.figure()
        gfg = sns.lineplot(data=df, x='time', y='value', hue='variable')
        gfg.set_ylim(bottom=0)

    @property
    def history(self):
        df = pd.DataFrame(
            {'time': self.time, 'consumption': self.c, 'capital': self.k, 'technology': self.A, 'labour': self.l})
        return df

    def production(self, A, K, N=1):
        return K ** (1 - self.alpha) * (A * N) ** self.alpha

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
        return np.log(c + 1e-9) + self.b * np.log(1 - l + 1e-9)

    def total_utility(self, c, l):
        pv_utility = self.utility(c, l) * (self.beta ** self.time)
        return np.sum(pv_utility)

    def model_step(self, k, a, l, savings_rate):
        y_ = self.production(a, k)
        k_ = y_ * savings_rate
        return y_, k_

    def model(self, c, l, s):
        for t in range(1, self.T):
            self.y[t], self.k[t] = self.model_step(self.k[t - 1], self.A[t - 1], l[t - 1], s[t - 1])

    def solve_closed_form(self):
        savings_rate = self.beta * (1 - self.alpha)
        consumption_rate = 1 - savings_rate
        for t in range(1, self.T):
            self.k[t] = self.production(self.A[t - 1], self.k[t - 1]) * savings_rate
        self.y[:] = self.A * self.k ** self.alpha
        self.c[:] = consumption_rate * self.y
        self.l[:] = self.alpha / (self.b * consumption_rate + self.alpha)
        self.r[:] = (1 - self.alpha) * (self.A * self.l / self.k) ** self.alpha + (
                1 - self.delta)  # the interest rate which ensures household assets = capital
        self.w[:] = self.alpha * self.A * (self.k / self.l) ** (1 - self.alpha)


if __name__ == "__main__":
    model = BrockMirman()
    model.solve()
    print()
