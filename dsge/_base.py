
class _BaseDSGE:
    """
    This is a base class for DSGE models. It ensures that models have a discount rate and number of periods.
    """
    def __init__(self, beta=0.9, T=10):
        self.beta = beta
        self.T= T