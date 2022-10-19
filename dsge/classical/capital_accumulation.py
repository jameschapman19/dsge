import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import grad, vmap
from scipy import optimize
from scipy.optimize import NonlinearConstraint

from dsge._base import _BaseDSGE


# matplotlib.use('TkAgg')


class CapitalAccumulation(_BaseDSGE):
    def __init__(self, A=1.0, T=10, delta=0.1, K_0=1.0, beta=0.9, solver='ls'):
        """
        Initializes the capital accumulation consumer problem

        References
        ----------
        .. [1] https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4171

        Parameters
        ----------
        A :  float
            Total Factor Productivity
        beta :
            Time Preference Factor
        T :
            Number of periods
        delta :
            Depreciation rate
        """
        super().__init__(beta, T, solver=solver)
        self.A = A
        self.delta = delta
        self.K_0 = K_0
        self.c = np.ones(self.T)
        self.k = np.zeros(self.T)

    @property
    def history(self):
        df = pd.DataFrame({'time': self.time, 'consumption': self.c, 'capital': self.k})
        return df

    def render(self):
        df = self.history
        df = pd.melt(df, id_vars=['time'], value_vars=['capital', 'consumption'])
        plt.figure()
        gfg = sns.lineplot(data=df, x='time', y='value', hue='variable')
        gfg.set_ylim(bottom=0)
        plt.show()

    def output(self, k):
        """
        Output of the firm
        Parameters
        ----------
        k : array_like
            Array of capital stock
        """
        return self.A * k

    def solve_least_squares(self):
        def euler(x):
            """
            Euler equation for consumption
            Parameters
            ----------
            x : array_like
                Array of consumption and capital stock
            """
            c = x[: self.T]
            self.model(c)
            consumption_euler = vmap(grad(self.utility))(c[:-1]) - self.beta * (
                    1 - self.delta + vmap(grad(self.output))(self.k[1:])
            ) * vmap(grad(self.utility))(c[1:])
            self.model(c)
            k_final = self.model_step(self.k[-1], c[-1])
            boundary = np.array(k_final, ndmin=1)
            return np.concatenate((consumption_euler, np.array(boundary, ndmin=1)))

        x0 = np.concatenate((self.c, self.k[1:]))
        self.solution = optimize.least_squares(euler, x0=x0, bounds=(0, np.inf))
        euler(self.solution.x)
        self.c = self.solution.x[: self.T]
        self.model(self.c)

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
            self.model(c)
            k_final = self.model_step(self.k[-1], c[-1])
            k_all = np.append(self.k, k_final)
            return k_all

        nlc = NonlinearConstraint(constraint, 0, np.inf)
        self.solution = optimize.minimize(
            negative_total_utility, x0=self.c, constraints=nlc, tol=1e-12
        )
        self.c = self.solution.x
        self.model(self.c)
        constraint(self.c)

    def utility(self, c):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return jnp.log(c)

    def total_utility(self, c):
        pv_utility = self.utility(c) * (self.beta ** self.time)
        return np.sum(pv_utility)

    def model_step(self, k, c):
        k_ = self.output(k) - c + (1 - self.delta) * k
        return k_

    def model(self, c):
        self.k[0] = self.K_0
        for t in range(1, self.T):
            self.k[t] = self.model_step(self.k[t - 1], c[t - 1])


if __name__ == "__main__":
    model = CapitalAccumulation(solver='min')
    model.solve()
    print(model.total_utility(model.c))
    model.render()
