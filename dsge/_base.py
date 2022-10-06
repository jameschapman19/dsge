from typing import Dict, Any, Optional

import numpy as np


class _BaseDSGE:
    """
    This is a base class for DSGE models. It ensures that models have a discount rate and number of periods.
    """
    metadata: Dict[str, Any] = {"render_modes": [None]}
    # define render_mode if your environment supports rendering
    render_mode: Optional[str] = None

    def __init__(self, beta=0.9, T=10, render_mode: Optional[str] = None):
        self.beta = beta
        self.T = T
        self.render_mode = self.render_mode
        self.t = np.arange(self.T)
