"""

Tests for the *montecarlosimulation* class.

Wrong parameter inputs are tested directly in the class.

"""

import sys
import pytest
from numpy.testing.utils import assert_array_almost_equal
from src.model_code.montecarlosimulation import MonteCarloSimulation


def test_montecarlosimulation_mspe_decomposition_same_as_mspe():
    """
    Test if the decomposition of the Mean Squared Prediction Error is
    approximately the same as the Mean Squared Prediction Error.

    As we choose for the test only 20 simulation iterations, we accept
    already deviations after the first decimal.

    """
    random_seeds_in = [1, 2, 3, 4]
    simulatiom_instance = (
        MonteCarloSimulation(n_repeat=20, random_seeds=random_seeds_in)
    )
    simulated_results = simulatiom_instance.calc_mse()
    mspe = simulated_results[0]
    added_mspe = (
        simulated_results[1] + simulated_results[2] + simulated_results[3]
    )

    # As the number of monte carlo iterations is kept small and the results are
    # based on simulations, we only look for the first decimal.
    assert_array_almost_equal(mspe, added_mspe, decimal=1)


if __name__ == '__main__':
    STATUS = pytest.main([sys.argv[1]])
    sys.exit(STATUS)
