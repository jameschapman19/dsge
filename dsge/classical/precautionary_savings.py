import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize
from scipy.optimize import NonlinearConstraint

from dsge._base import _BaseDSGE


class PrecautionarySavings(_BaseDSGE):
    """
    References
    ----------
    .. [1] https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4171
    """

    def __init__(self, W_0=1.0, beta=1.0, T=10, T_shock=5, W_shock=0.5, eps=1e-3, solver='min'):
        """

        Parameters
        ----------
        W_0 : float
            Initial wage
        beta : float
            Discount factor
        T : int
            Number of periods
        T_shock : int
            Period of shock
        W_shock : float
            Shock to wage
        eps : float
            Small number to avoid log(0)
        solver : str
            Solver to use. Either 'ls' for least squares or 'min' for minimize
        """
        super().__init__(beta, T, solver=solver)
        self.W_0 = W_0
        self.T_shock = T_shock
        self.W_shock = W_shock
        self.eps = eps
        self.history_vars = ['consumption', 'wage', 'savings']
        self.c = np.zeros(self.T)
        self.s = np.zeros(self.T)
        self.w = np.ones(self.T) * self.W_0
        self.w[self.T_shock:] *= self.W_shock

    @property
    def history(self):
        df = pd.DataFrame({'time': self.time, 'consumption': self.c, 'wage': self.w, 'savings': self.s})
        return df

    def render(self):
        df = self.history
        df = pd.melt(df, id_vars=['time'], value_vars=self.history_vars)
        plt.figure()
        gfg = sns.lineplot(data=df, x='time', y='value', hue='variable')
        gfg.set_ylim(bottom=0)

    def solve_minimize(self):
        def negative_total_utility(c):
            """
            Negative total utility function for consumption c
            Parameters
            ----------
            c : float
                Consumption
            """
            self.model(c)
            return -self.total_utility(c)

        def constraint(c):
            self.model(c)
            s_final, _ = self.model_step(c[-1], self.s[-1], self.w[-1])
            s_all = np.insert(self.s, 0, s_final)
            return s_all

        nlc = NonlinearConstraint(constraint, 0, np.inf)
        solution = optimize.minimize(negative_total_utility, np.ones(self.T), constraints=nlc)
        self.c = solution.x
        self.model(self.c)

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
        pv_utility = self.utility(c) * (self.beta ** self.time)
        return np.sum(pv_utility)

    def model_step(self, c, s, w):
        s_ = s + w - c
        if self.t == (self.T_shock):
            w_ = self.W_shock
        else:
            w_ = w
        return s_, w_

    def model(self, c):
        self.t = 1
        for t in range(1, self.T):
            self.s[t], self.w[t] = self.model_step(c[t - 1], self.s[t - 1], self.w[t - 1])
            self.t += 1


if __name__ == "__main__":
    model = PrecautionarySavings()
    model.solve()
    print(model.total_utility(model.c))
    model.render()
    plt.show()
