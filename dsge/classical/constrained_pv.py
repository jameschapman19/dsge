import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import grad, vmap
from scipy import optimize
from scipy.optimize import NonlinearConstraint

from dsge._base import _BaseDSGE


class ConstrainedPV(_BaseDSGE):
    """
    References
    ----------
    .. [1] https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4171
    """

    def __init__(self, W=1.0, R=1.0, beta=0.5, T=10, eps=1e-3, solver='ls'):
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
        super().__init__(beta, T, solver=solver)
        self.W = W
        self.R = R
        self.eps = eps
        self.c = np.ones(self.T)
        self.w = np.ones(self.T) * self.W

    @property
    def history(self):
        df = pd.DataFrame({'time': self.time, 'consumption': self.c})
        return df

    def render(self):
        df = self.history
        df = pd.melt(df, id_vars=['time'], value_vars=['consumption'])
        plt.figure()
        gfg = sns.lineplot(data=df, x='time', y='value', hue='variable')
        gfg.set_ylim(bottom=0)

    def solve_least_squares(self):
        def euler(c):
            """
            Euler equation for consumption
            """
            euler = vmap(grad(self.utility))(c[:-1]) - self.beta * self.R * vmap(grad(self.utility))(c[1:])
            return np.concatenate((euler, self.budget(c)))

        solution = optimize.least_squares(euler, x0=np.zeros(self.T), bounds=(0, np.inf), max_nfev=10000)
        self.c[:] = solution.x

    def solve_minimize(self):
        def negative_total_utility(c):
            """
            Negative total utility function for consumption c
            Parameters
            ----------
            c : float
                Consumption
            """
            return -self.total_utility(c)

        def constraint(c):
            """
            Constraint
            """
            return -self.budget(c)

        bounds = [(0, None) for i in range(len(self.c))]
        nlc = NonlinearConstraint(constraint, 0, np.inf)
        solution = optimize.minimize(negative_total_utility, x0=np.zeros(self.T), bounds=bounds, constraints=nlc)
        self.c[:] = solution.x

    def budget(self, c):
        """
        Budget constraint
        """
        Rt = self.R ** (1 - (np.arange(self.T) + 1))
        return np.array([np.dot(c, Rt) - self.W])

    def utility(self, c):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return jnp.log(c + self.eps)

    def total_utility(self, c):
        pv_utility = self.utility(c) * (self.beta ** self.time)
        return np.sum(pv_utility)

    def model_step(self, t, w, c):
        w_ = w - self.R ** (1 - (t + 1)) * c
        return w_

    def model(self, c):
        for t in range(1, self.T):
            self.w[t] = self.model_step(t, self.w[t - 1], c)


if __name__ == '__main__':
    model = ConstrainedPV()
    model.solve()
    model.render()
    plt.show()
