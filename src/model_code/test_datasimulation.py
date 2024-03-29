"""

Tests for the *DataSimulation* class.

Wrong parameter inputs are tested directly in the class.

"""

import sys
import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal
from src.model_code.datasimulation import DataSimulation


def test_datasimulation_if_same_if_new_instance_with_indicator():
    """
    Test if the *DataSimulation* class is truly deterministic if
    the random seed is specified.

    """
    first_data_set = DataSimulation(random_seed=1).indicator_model()
    second_data_set = DataSimulation(random_seed=1).indicator_model()

    assert_array_almost_equal(first_data_set[0], second_data_set[0])
    assert_array_almost_equal(first_data_set[1], second_data_set[1])


def test_datasimulation_indicator_function():
    """ Test the indicator function. """
    indicator_out = (
        DataSimulation(n_size=10)._indicator_function(np.ones((10,)), 1)
    )
    assert_array_almost_equal(indicator_out, np.ones(10))


def test_datasimulation_friedman_shape():
    """ Test the output shape for the friedman #1 model. """
    data_set = DataSimulation(random_seed=1, n_size=100).friedman_1_model()
    assert data_set[0].shape == (100, 10)
    assert data_set[1].shape == (100,)


if __name__ == '__main__':
    STATUS = pytest.main([sys.argv[1]])
    sys.exit(STATUS)
