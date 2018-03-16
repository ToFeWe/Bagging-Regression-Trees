"""

Tests for the module *calc_finite_sample*.

"""

import sys
import pytest
import numpy as np
from src.analysis.theory_simulation.calc_finite_sample import bagged_indicator, indicator

# No need for fixtures here as the setup is so simple and there are only two
# functions.


def test_indicator():
    """
    A function to test if the unbagged indicator function
    is specified correctly.

    """
    assert indicator(0, 1) == 0
    assert indicator(1, 0) == 1


def test_bagged_indicator_ones():
    """
    A function to test if the bagged indicator function
    is specified correctly.
    We test here with ones.

    """
    sample = np.ones(10)
    b_iterations = 10
    x_value = 0
    assert bagged_indicator(x_value, sample, b_iterations) == 0


def test_bagged_indicator_zeros():
    """
    A function to test if the bagged indicator function
    is specified correctly.
    We test here with zeros.

    """

    sample = np.zeros(10)
    b_iterations = 10
    x_value = 0
    assert bagged_indicator(x_value, sample, b_iterations) == 1


if __name__ == '__main__':
    STATUS = pytest.main([sys.argv[1]])
    sys.exit(STATUS)
