import numpy as np
from scipy import optimize

from dsge.classical.rbc import RBC


class SimpleBrockMirman(RBC):
    """
    References
    ----------
    .. [1] http://www.econ2.jhu.edu/people/ccarroll/public/LectureNotes/DSGEModels/BrockMirman/
    """

    def __init__(self, alpha=0.5, beta=0.5, K_0=1, A_0=1, T=10, G=0.02, solution='closed_form'):
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
        super().__init__(alpha=alpha, beta=beta, T=T, delta=1, A_0=A_0, K_0=K_0, G=G)
        self.solution = solution

    def utility(self, c, **kwargs):
        return np.log(c + 1e-9)

    def production(self, A, K, **kwargs):
        return A * K ** self.alpha

    def solve(self):
        """
        Solves the constrained consumer problem
        """
        if self.solution == 'closed_form':
            self.solve_closed_form()
        else:
            raise NotImplementedError

    def solve_closed_form(self):
        kappa = 1 - self.alpha * self.beta
        for t in range(1, self.T):
            self.K[t] = self.alpha * self.beta * self.A[t - 1] * self.K[t - 1] ** self.alpha
        self.Y[:] = self.A * self.K ** self.alpha
        self.C[:] = kappa * self.Y


if __name__ == "__main__":
    model = SimpleBrockMirman()
    model.solve()
    print()
