"""

A module to calculate the results for the general stump predictor in Subsection
4.2 (Theorem 4.1) of the paper for the dynamic environment of x.

Replacing the bootstrap procedure by a subsampling scheme, we can here calculate
upper bounds for the Variance and the Bias of stump predictors seen in 
subsection 4.2 and following the framework developed by :cite:`Buhlmann2002`.

"""

import numpy as np
from scipy.stats import norm

import json
import pickle

from bld.project_paths import project_paths_join as ppj


def bias_normal_splits(c, a, gamma):
    """
    Calculates the squared bias for stump predictors as defined in the paper in
    Theorem 4.1.

    Parameters
    ----------

    c: int, float
        The gridpoint to be considered.

    a: float
        The subsampling fraction.

    gamma: float
        The rate of convergence of the estimator.

    Returns the squared bias.
    """
    bias = (norm.cdf(c * a ** gamma) - norm.cdf(c)) ** 2
    return bias


def variance_normal_splits(c, a, gamma):
    """
    Calculates the variance for stump predictors as defined in the paper in
    Theorem 4.1.

    Parameters
    ----------

    c: int, float
        The gridpoint to be considered.

    a: float
        The subsampling fraction.

    gamma: float
        The rate of convergence of the estimator.

    Returns the variance.
    """
    variance = (a * norm.cdf(c * a ** gamma) * (1 - norm.cdf(c * a ** gamma)))
    return variance


def calculate_normal_splits(settings):
    """
    Calculate the Bias and the Variance for the case of subagging based on the
    calculation settings defined in *settings*.

    settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the calculation set-up that is specific to the 
        stump predictor simulation.

    Returns the calculated values as a dictionary.

    """

    output = {}

    # Create a range of c values that we will iterate over for each subsampling
    # fraction a and save it to output dictionary for plotting.
    c_range = np.linspace(
        settings['c_min'],
        settings['c_max'],
        num=settings['c_gridpoints'])
    output['c_range'] = c_range

    # Create the array with the a_range. The formation is chosen this way, for
    # plotting reasons.
    # Note that the first a is always one as a reference (unbagged). Hence no
    # fraction.
    a_range = np.array([
        settings['a_array']['first_a'],
        settings['a_array']['second_a'][0] /
        settings['a_array']['second_a'][1],
        settings['a_array']['third_a'][0] /
        settings['a_array']['third_a'][1],
        settings['a_array']['fourth_a'][0] /
        settings['a_array']['fourth_a'][1]
    ])

    # Loop over the range of c values.
    for i_a, a in enumerate(a_range):
        # For the list of a values (subsampling fraction).

        # Create an array that save the results for Bias and Variance
        bias_array = np.ones(settings['c_gridpoints']) * np.nan
        var_array = np.ones(settings['c_gridpoints']) * np.nan

        for i_c, c in enumerate(c_range):
            # The calculation are done straight forward following the
            # derivations in the paper.
            bias_array[i_c] = bias_normal_splits(c, a, settings['gamma'])
            var_array[i_c] = variance_normal_splits(c, a, settings['gamma'])

        mse_array = np.add(bias_array, var_array)

        # Save the results to the dictonary. Note that we use the iteration number
        # as the key, since we follow a similar logic in the plotting part.
        output[i_a] = {}
        output[i_a]['bias'] = bias_array
        output[i_a]['variance'] = var_array
        output[i_a]['mse'] = mse_array
    return output


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS", "normal_splits_settings.json")) as f:
        normal_splits_settings_imported = json.load(f)

    calculate_normal_splits = calculate_normal_splits(
        normal_splits_settings_imported)

    with open(ppj("OUT_ANALYSIS_THEORY", "output_normal_splits.pickle"), "wb") as out_file:
        pickle.dump(calculate_normal_splits, out_file)
