"""

Tests for the module *calc_toy_example.

"""

import sys
import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
from src.analysis.theory_simulation.calc_toy_example import *


@pytest.fixture
def setup():
    c_values = [0.0, np.inf, -np.inf, 0.1, -0.1]
    return c_values

def test_convolution_cdf_df(setup):
    out = setup
    assert_array_almost_equal(convolution_cdf_df(out[0]),
                              0.5)
    assert_array_almost_equal(convolution_cdf_df(out[1]),
                              1)
    assert_array_almost_equal(convolution_cdf_df(out[2]),
                              0)
    
def test_convolution_cdf_squared_df(setup):
    out = setup
    assert_array_almost_equal(convolution_cdf_squared_df(out[0]),
                              1/3)
    assert_array_almost_equal(convolution_cdf_squared_df(out[1]),
                              1)
    assert_array_almost_equal(convolution_cdf_squared_df(out[2]),
                              0)

def test_calculate_bias_bagged(setup):
    out = setup
    assert_array_almost_equal(calculate_bias_bagged(out[0]),
                              0)
    assert_array_almost_equal(calculate_bias_bagged(out[1]),
                              0)
    assert_array_almost_equal(calculate_bias_bagged(out[2]),
                              0)
    assert_array_almost_equal(calculate_bias_bagged(out[3]), 
                              calculate_bias_bagged(out[4]))
    # Bias is greater than zero for c values close to zero.
    assert calculate_bias_bagged(out[3]) > 0

def test_calculate_var_bagged(setup):
    out = setup
    assert_array_almost_equal(calculate_var_bagged(out[1]),
                              0)
    assert_array_almost_equal(calculate_var_bagged(out[2]),
                              0)
    assert_array_almost_equal(calculate_var_bagged(out[3]), 
                              calculate_var_bagged(out[4]))
    # Variance is the greatest at zero.
    assert calculate_var_bagged(out[0]) > calculate_var_bagged(out[3])
    
def test_calculate_var_unbagged(setup):
    out = setup
    assert_array_almost_equal(calculate_var_unbagged(out[1]),
                              0)
    assert_array_almost_equal(calculate_var_unbagged(out[2]),
                              0)
    assert calculate_var_unbagged(out[0]) == 0.25
    assert_array_almost_equal(calculate_var_unbagged(out[3]), 
                              calculate_var_unbagged(out[4]))

    
if __name__ == '__main__':
    status = pytest.main([sys.argv[1]])
    sys.exit(0)
 