import numpy as np
from scipy import optimize


class ConsumerCapitalAccumulation:
    def __init__(self, A=1.0, Beta=0.5, T=10, delta=0.1, k_init=1.0):
        """
        Initializes the capital accumulation consumer problem
        References
        ----------
        .. [1] https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4171
        Parameters
        ----------
        A :  float
            Total Factor Productivity
        Beta :
            Time Preference Factor
        T :
            Number of periods
        delta :
            Depreciation rate
        k1 :
            Initial capital stock
        solution_method :
            Solution method for the constrained problem
        """
        self.A = A
        self.Beta = Beta
        self.T = T
        self.delta = delta
        self.k_init = k_init

    def output(self, k):
        """
        Output of the firm
        Parameters
        ----------
        k : array_like
            Array of capital stock
        """
        return self.A * k

    def output_grad(self, k):
        """
        Gradient of output function
        Parameters
        ----------
        k : array_like
            Array of capital stock
        """
        return self.A

    def solve(self):
        c0 = np.ones(self.T)
        k0 = np.zeros(self.T - 1)
        x0 = np.concatenate((c0, k0))
        self.solution = optimize.least_squares(self.euler, x0=x0, bounds=(0, np.inf))
        self.c = self.solution.x[: self.T]
        self.k = np.concatenate((np.array([self.k1]), self.solution.x[self.T :]))

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
        return 1 / c

    def euler(self, x):
        """
        Euler equation for consumption
        Parameters
        ----------
        x : array_like
            Array of consumption and capital stock
        """
        c = x[: self.T]
        k = np.concatenate((np.array([self.k_init]), x[self.T :]))
        consumption_euler = self.utility_grad(c[:-1]) - self.Beta * (
            1 - self.delta + self.output_grad(k[1:])
        ) * self.utility_grad(c[1:])
        capital_euler = k[1:] - self.output(k[:-1]) + c[:-1] - (1 - self.delta) * k[:-1]
        boundary_condition = np.array(
            [-self.output(k[-1]) + c[-1] - (1 - self.delta) * k[-1]]
        )
        return np.concatenate((consumption_euler, capital_euler, boundary_condition))


if __name__ == "__main__":
    model = ConsumerCapitalAccumulation()
    model.solve()
    print()
