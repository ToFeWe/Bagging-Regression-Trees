# -*- coding: utf-8 -*-
"""

A module to calculate the results for the introductory example in Subsection
3.2 of the paper for the dynamic environment of x.

Given the choice of the appropriate environment of x, the estimator does not
stabilizes even asymptotically and we can illustrate the effects of bagging on
it.

"""
import pickle
import json
import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

from bld.project_paths import project_paths_join as ppj


def convolution_cdf_df(c_value):
    """ Calculate the convolution as defined by :cite:`Buhlmann2002` and as
    used in the introductory example of our paper for the the c_value.d.f of the
    standard normal distribution and the standard normal density for the
    gridpoint *c_value* for the real number line.

    Parameters
    ----------
    c_value: float, int
        The gridpoint to be considered.

    """
    # We use the lambda operator here, as its a very simple function, we want
    # to integrate over and its only used once.
    convolution = integrate.quad(
        lambda y: norm.cdf(
            c_value - y) * norm.pdf(y), -np.inf, np.inf)[0]
    return convolution


def convolution_cdf_squared_df(c_value):
    """ Calculate the convolution as defined by :cite:`Buhlmann2002` and as
    used in the introductory example of our paper for the the squared c_value.d.f of
    the standard normal distribution and the standard normal density for the
    gridpoint *c_value* for the real number line.

    Parameters
    ----------
    c_value: float, int
        The gridpoint to be considered.

    """
    # We use the lambda operator here, as its a very simple function, we want
    # to integrate over and its only used once.
    convolution = integrate.quad(lambda y: norm.cdf(
        c_value - y) ** 2 * norm.pdf(y), -np.inf, np.inf)[0]
    return convolution


def calculate_bias_bagged(c_value):
    """ Calculate the squared bias for the bagged predictor given the grid point
    *c_value*.

    Parameters
    ----------
    c_value: float, int
        The gridpoint to be considered.

    """
    bias_bagged = (convolution_cdf_df(c_value) - norm.cdf(c_value)) ** 2
    return bias_bagged


def calculate_var_bagged(c_value):
    """ Calculate the variance for the bagged predictor given the grid point
    *c_value*.

    Parameters
    ----------
    c_value: float, int
        The gridpoint to be considered.

    """
    var_bagged = (
        convolution_cdf_squared_df(c_value) - convolution_cdf_df(c_value) ** 2
    )
    return var_bagged


def calculate_var_unbagged(c_value):
    """ Calculate the variance for the bagged predictor given the grid point
    *c_value*.

    Parameters
    ----------
    c_value: float, int
        The gridpoint to be considered.


    """
    var_unbagged = norm.cdf(c_value) * (1 - norm.cdf(c_value))
    return var_unbagged


def calculate_toy_example(settings):
    """
    Calculate the Bias and the Variance for the case of bagged and unbagged
    predictor based on the calulation settings defined in *settings*.

    settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the calculation set-up that is specific to the
        introductory simulation.

    Returns the calculated values as a dictionary.

    """
    # Create grid with *c_value* values that we want to consider.
    c_range = np.linspace(
        settings['c_min'],
        settings['c_max'],
        num=settings['c_gridpoints'])

    # Create the arrays that will be used to save the results.
    bagged_var = np.ones(settings['c_gridpoints']) * np.nan
    unbagged_var = np.ones(settings['c_gridpoints']) * np.nan
    bagged_bias = np.ones(settings['c_gridpoints']) * np.nan

    # We save all results to the dictonary *output*.
    output = {}
    output['c_range'] = c_range

    # Loop over *c_value* values that we want to consider and save the results.
    # Note that the unbagged predcitor is unbiased.
    for i_c, c_value in enumerate(c_range):
        bagged_var[i_c] = calculate_var_bagged(c_value)
        unbagged_var[i_c] = calculate_var_unbagged(c_value)
        bagged_bias[i_c] = calculate_bias_bagged(c_value)

    output['bagged'] = {}
    output['bagged']['variance'] = bagged_var
    output['bagged']['bias'] = bagged_bias

    output['unbagged'] = {}
    output['unbagged']['variance'] = unbagged_var
    # For plotting reasons we also save squared bias of the unbagged predictor
    # which is zero by defintion.
    output['unbagged']['bias'] = np.zeros(settings['c_gridpoints'])

    return output


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS", "toy_example_settings.json")) as f:
        TOY_EXAMPLE_SETTINGS_IMPORTED = json.load(f)

    CALCULATE_TOY_EXAMPLE = calculate_toy_example(
        TOY_EXAMPLE_SETTINGS_IMPORTED)

    with open(ppj("OUT_ANALYSIS_THEORY", "output_toy_example.pickle"), "wb") as out_file:
        pickle.dump(CALCULATE_TOY_EXAMPLE, out_file)
