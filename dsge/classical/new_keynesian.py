import numpy as np
from scipy import optimize


class NewKeynesian:
    def __init__(self, A=1.0, Beta=0.5, T=10, delta=0.1, k_init=1.0, sigma=0.5, eta=0.5, psi=0.5, phi=0):
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
        self.sigma=sigma
        self.eta=eta
        self.psi=psi
        self.A = A
        self.Beta = Beta
        self.T = T
        self.delta = delta
        self.k_init = k_init
        self.phi=phi

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

    def utility(self, C,N):
        """
        Utility function for consumption c
        Parameters
        ----------
        c : float
            Consumption
        """
        return C**(1-self.sigma)/(1-self.sigma)-self.psi*N**(1+self.eta)/(1+self.eta)

    def update_price():
        if self.phi==0:
            P_hash=self.eps/(self.eps-1)*W/self.A
        P=(1-self.phi)*P_hash**(1-self.eps)+self.phi*P**(1-self.eps)
        return P**(1/(1-self.eps))

    def euler(self, x):
        """
        Euler equation for consumption
        Parameters
        ----------
        x : array_like
            Array of consumption and capital stock
        """
        pass


if __name__ == "__main__":
    model = NewKeynesian()
    model.solve()
    print()
