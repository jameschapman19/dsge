from dsge.classical.rbc import RBC


class BrockMirman(RBC):
    """
    References
    ----------
    .. [1] https://personal.lse.ac.uk/vernazza/_private/RBC%20Models.pdf
    """

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=100, G=0.02,b=0.5, solution='closed_form'):
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
        self.solution = solution

    def solve(self):
        """
        Solves the constrained consumer problem
        """
        if self.solution == 'closed_form':
            self.solve_closed_form()
        else:
            raise NotImplementedError

    def solve_closed_form(self):
        self.savings_rate = self.beta * (1 - self.alpha)
        self.consumption_rate = 1 - self.savings_rate
        for t in range(1, self.T):
            self.K[t] = self.production(self.A[t - 1], self.K[t - 1]) * self.savings_rate
        self.Y[:] = self.A * self.K ** self.alpha
        self.C[:] = self.consumption_rate * self.Y
        self.N = self.alpha / (self.b * self.consumption_rate + self.alpha)
        self.R = (1-self.alpha)*(self.A*self.N/self.K)**self.alpha+(1-self.delta) # the interest rate which ensures household assets = capital
        self.W = self.alpha*self.A*(self.K/self.N)**(1-self.alpha)


if __name__ == "__main__":
    model = BrockMirman()
    model.solve()
    print()
