import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax import grad
from scipy import optimize
from scipy.optimize import NonlinearConstraint

from dsge._base import _BaseDSGE


class TFPShock(_BaseDSGE):
    def __init__(self, beta=0.96, T=30, alpha=0.35, delta=0.06, rho=0.8, A_0=1.05, A_bar=1, A_eps=0, solver='min'):
        """
        Initializes the capital accumulation consumer problem

        References
        ----------
        .. [1] https://www.sciencedirect.com/science/article/pii/S1477388020300244

        Parameters
        ----------
        beta: float
            Discount factor
        T: int
            Number of periods
        alpha: float
            Output-capital elasticity
        delta: float
            Depreciation rate
        rho: float
            Persistence of TFP shock
        A_0: float
            Initial TFP
        A_bar: float
            Steady state TFP
        A_eps: float
            Standard deviation of TFP shock
        solver: str
            Solver to use
        """
        super().__init__(beta, T, solver=solver)
        self.rho = rho
        self.alpha = alpha
        self.delta = delta
        self.A_bar = A_bar
        self.A_eps = A_eps
        self.A_0 = A_0
        self.R_bar = (1 - beta + beta * delta) / beta  # steady state interest rate
        self.K_bar = ((1 - beta + beta * delta) / (alpha * self.A_bar * beta)) ** (
                    1 / (alpha - 1))  # steady state capital
        self.Y_bar = self.A_bar * self.K_bar ** alpha  # steady state output
        self.I_bar = self.delta * self.K_bar  # steady state investment
        self.C_bar = self.Y_bar - self.I_bar  # steady state consumption
        self.k = np.zeros(self.T)
        self.i = np.zeros(self.T)
        self.y = np.zeros(self.T)
        self.c = np.ones(self.T) * self.C_bar
        self.a = np.ones(self.T) * self.A_0
        self.k[0] = self.capital_growth(self.I_bar, self.K_bar)
        self.y[0] = self.production(self.k[0], self.a[0])

    @property
    def history(self):
        df = pd.DataFrame(
            {'time': self.time, 'consumption': self.c, 'investment': self.i, 'output': self.y,
             'capital': self.k, 'tfp': self.a})
        return df

    def render(self):
        df = self.history
        value_vars = ['capital', 'consumption', 'investment', 'output', 'tfp']
        fig, axs = plt.subplots(len(value_vars), 1, figsize=(12, 8))
        for ax, var in zip(axs, value_vars):
            sns.lineplot(data=df, x='time', y=var, ax=ax)

    def tfp_shock(self, A):
        return A ** self.rho + np.random.normal() * self.A_eps

    def solve_minimize(self):
        def function(c):
            return -self.total_utility(c)

        def constraint(c):
            self.model(c)
            _, _, k_final, _ = self.model_step(self.y[-1], self.i[-1], self.k[-1], c[-1], self.a[-1])
            return k_final - self.K_bar

        bounds = [(0, None) for _ in range(self.T)]
        nlc = NonlinearConstraint(constraint, 0, 1)
        self.solution = optimize.minimize(function, np.ones(self.T) * self.C_bar, jac=grad(function), method='SLSQP',
                                          constraints=nlc, bounds=bounds)
        self.c = self.solution.x
        constraint(self.c)
        assert self.solution.success

    def production(self, k, A):
        """
        Production function
        """
        if k > 0:
            return A * k ** self.alpha
        else:
            return 0

    def capital_growth(self, I, K):
        """
        Capital growth function
        """
        return I + (1 - self.delta) * K

    def investment(self, Y, C):
        """
        Investment function
        """
        return Y - C

    def model_step(self, y, i, k, c, a):
        k_ = self.capital_growth(i, k)
        a_ = self.tfp_shock(a)
        y_ = self.production(k_, a_)
        i_ = self.investment(y_, c)
        return y_, i_, k_, a_

    def model(self, c):
        self.i[0] = self.investment(self.y[0], c[0])
        for t in range(1, self.T):
            self.y[t], self.i[t], self.k[t], self.a[t] = self.model_step(self.y[t - 1], self.i[t - 1], self.k[t - 1],
                                                                         c[t],
                                                                         self.a[t - 1])

    def utility(self, c):
        """
        Utility function for consumption c
        """
        return jnp.log(c + 1e-9)

    def total_utility(self, c):
        pv_utility = self.utility(c) * (self.beta ** self.time)
        return jnp.sum(pv_utility)


if __name__ == "__main__":
    model = TFPShock()
    model.solve()
    print(model.total_utility(model.c))
    model.render()
    plt.show()
