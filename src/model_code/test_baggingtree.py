"""

Tests for the *BaggingTree* class.

Wrong parameter inputs are tested directly in the class.

"""

import sys
import pytest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_raises
from src.model_code.baggingtree import BaggingTree


@pytest.fixture
def setup():
    """ Set up the test fixture. """
    np.random.seed(1)
    x_matrix_train = np.random.normal(size=(10, 2))
    y_vector_train = (
        x_matrix_train[:, 0] +
        x_matrix_train[:, 1] + np.random.normal(size=10)
    )
    y_vector_test = np.ones((10, 2))
    return x_matrix_train, y_vector_train, y_vector_test


def test_baggingtree_if_same_if_new_instance(setup):
    """
    Test if the bagging algorithm gives us deterministic results. If the
    random seed is the same for two new instances.

    """
    x_matrix_train, y_vector_train, y_vector_test = setup
    bagged_tree = BaggingTree(random_seed=1, b_iterations=5)
    first_prediction = (
        bagged_tree.fit(x_matrix_train, y_vector_train).predict(y_vector_test)
    )
    bagged_tree = BaggingTree(random_seed=1, b_iterations=5)
    second_prediction = (
        bagged_tree.fit(x_matrix_train, y_vector_train).predict(y_vector_test)
    )
    assert_array_almost_equal(first_prediction, second_prediction)


def test_baggingtree_if_different_if_same_instance(setup):
    """
    Test if the bagging algorithm gives us deterministic results.
    The results should differ for two fitting processes with the
    same instance.

    """

    x_matrix_train, y_vector_train, y_vector_test = setup
    bagged_tree = BaggingTree(random_seed=1, b_iterations=10)
    first_prediction = (
        bagged_tree.fit(x_matrix_train, y_vector_train).predict(y_vector_test)
    )
    second_prediction = (
        bagged_tree.fit(x_matrix_train, y_vector_train).predict(y_vector_test)
    )
    # We check if they are not equal.
    # The probability that they are equal is so low that we should be worried
    # else wise.
    assert_raises(
        AssertionError,
        assert_array_almost_equal,
        first_prediction,
        second_prediction
    )


def test_baggingtree_two_predicts_the_same(setup):
    """
    Test if the bagging algorithm gives us deterministic results.
    If we make a predict twice, the predicted values should be the
    same.

    """

    x_matrix_train, y_vector_train, y_vector_test = setup
    bagged_tree = BaggingTree(random_seed=1, b_iterations=5)
    fitted_tree = bagged_tree.fit(x_matrix_train, y_vector_train)
    first_predict = fitted_tree.predict(y_vector_test)
    second_predict = fitted_tree.predict(y_vector_test)
    assert_array_almost_equal(first_predict, second_predict)


def test_baggingtree_with_zeros_and_ones():
    """ Test the bagging algorithm with zeros and ones. """
    bagged_tree = BaggingTree(random_seed=1, b_iterations=5)
    fitted_tree = bagged_tree.fit(np.zeros((10, 2)), np.zeros(10))
    prediction = fitted_tree.predict(np.ones((10, 2)))
    assert_array_almost_equal(prediction, np.zeros(10))


if __name__ == '__main__':
    STATUS = pytest.main([sys.argv[1]])
    sys.exit(STATUS)
