import matplotlib.pyplot as plt

from dsge.classical.precautionary_savings import PrecautionarySavings


class AmbiguousPrecautionarySavings(PrecautionarySavings):
    """
    References
    ----------
    .. [1] https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4171
    """

    def __init__(self, W_0=1.0, beta=1.0, T=10, T_shock=5, W_shock=0.5, eps=1e-3, solver='min'):
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
        super().__init__(W_0, beta, T, T_shock, W_shock, eps, solver)

    def solve_least_squares(self):
        raise NotImplementedError


if __name__ == "__main__":
    model = PrecautionarySavings()
    model.solve()
    print(model.total_utility(model.c))
    model.render()
    plt.show()
