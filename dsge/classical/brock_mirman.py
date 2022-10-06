import numpy as np

from dsge.classical._rbc import _RBC


class BrockMirman(_RBC):
    """
    References
    ----------
    .. [1] https://personal.lse.ac.uk/vernazza/_private/RBC%20Models.pdf
    """

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=100, G=0.02, b=0.5):
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
        super().__init__(alpha=alpha, beta=beta, T=T, delta=1, A_0=A_0, K_0=K_0, G=G, b=b)

    def render(self, mode="human"):
        pass

    def solve(self):
        """
        Solves the constrained consumer problem
        """
        self.solve_closed_form()

    def solve_closed_form(self):
        self.savings_rate = self.beta * (1 - self.alpha)
        self.consumption_rate = 1 - self.savings_rate
        for t in range(1, self.T):
            self.k[t] = self.production(self.A[t - 1], self.k[t - 1]) * self.savings_rate
        self.y[:] = self.A * self.k ** self.alpha
        self.c[:] = self.consumption_rate * self.y
        self.l = self.alpha / (self.b * self.consumption_rate + self.alpha)
        self.r = (1 - self.alpha) * (self.A * self.l / self.k) ** self.alpha + (
                1 - self.delta)  # the interest rate which ensures household assets = capital
        self.w = self.alpha * self.A * (self.k / self.l) ** (1 - self.alpha)

    def total_utility(self, c, l):
        pv_utility = self.utility(c, l=l) * (self.beta ** self.t)
        return np.sum(pv_utility)


if __name__ == "__main__":
    model = BrockMirman()
    model.solve()
