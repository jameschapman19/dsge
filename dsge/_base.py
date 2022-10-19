from abc import abstractmethod
from typing import Dict, Any, Optional

import numpy as np


# matplotlib.use('TkAgg')

class _BaseDSGE:
    """
    This is a base class for DSGE models. It ensures that models have a discount rate and number of periods.
    """
    metadata: Dict[str, Any] = {"render_modes": [None]}
    # define render_mode if your environment supports rendering
    render_mode: Optional[str] = None

    def __init__(self, beta=0.9, T=10, render_mode: Optional[str] = None, solver='ls'):
        self.beta = beta
        self.T = T
        self.render_mode = self.render_mode
        self.time = np.arange(self.T)
        self.t = 0
        self.solver = solver

    @property
    @abstractmethod
    def history(self):
        pass

    def utility(self, **args):
        raise NotImplementedError

    @abstractmethod
    def total_utility(self, **args):
        raise NotImplementedError

    def solve(self):
        if self.solver == 'ls':
            self.solve_least_squares()
        elif self.solver == 'min':
            self.solve_minimize()
        else:
            raise ValueError('Solver not recognized')

    @abstractmethod
    def solve_least_squares(self):
        raise NotImplementedError

    @abstractmethod
    def solve_minimize(self):
        raise NotImplementedError

    def model_step(self, *args):
        raise NotImplementedError

    def model(self, *args):
        raise NotImplementedError

    def step_time(self):
        self.t += 1
        if self.t >= self.T:
            done = True
        else:
            done = False
        return done
