"""

Tests for the module *calc_normal_splits*.

"""

import sys
import pytest
import numpy as np
from src.analysis.theory_simulation.calc_normal_splits import \
    bias_normal_splits,\
    variance_normal_splits

# No need for fixtures here as the setup is so simple and there are only two
# functions.


def test_bias_normal_splits():
    """
    A function to test the squared-bias of the stump predictor.
    We use three parameter specifications for which we can derive
    clear theoretical predictions, which are then testable.

    """
    assert bias_normal_splits(0, 0, 1 / 2) == 0
    assert bias_normal_splits(1, 1, 1 / 2) == 0
    assert bias_normal_splits(1, 0, 1 / 2) > 0


def test_variance_normal_splits():
    """
    A function to test the variance of the stump predictor.
    We use two parameter specifications for which we can derive
    clear theoretical predictions, which are then testable.

    """

    assert variance_normal_splits(0, 1, 1 / 2) == 0.25
    assert variance_normal_splits(np.inf, 1, 1 / 2) == 0


if __name__ == '__main__':
    STATUS = pytest.main([sys.argv[1]])
    sys.exit(STATUS)
