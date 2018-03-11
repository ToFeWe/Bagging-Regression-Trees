"""

Tests for the module *calc_normal_splits*.

"""

import sys
import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
from src.analysis.theory_simulation.calc_normal_splits import *

# No need for fixtures here as the setup is so simple and there are only two
# functions.
def test_bias_normal_splits():
    assert bias_normal_splits(0,0,1/2) == 0
    assert bias_normal_splits(1,1,1/2) == 0
    assert bias_normal_splits(1,0,1/2) > 0

def test_variance_normal_splits():
    assert variance_normal_splits(0,1,1/2) == 0.25
    assert variance_normal_splits(np.inf,1,1/2) == 0


if __name__ == '__main__':
    status = pytest.main([sys.argv[1]])
    sys.exit(0)
