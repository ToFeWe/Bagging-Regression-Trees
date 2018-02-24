# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:52:10 2017

@author: gl1rk
"""

import numpy as np
from scipy.stats import norm
import pickle
import scipy.integrate as integrate

import json
from bld.project_paths import project_paths_join as ppj






def convolution_cdf_df(c):
    ''' Calculate the convolution as defined by Bühlmann and Yu (2002) and as
    used in the inductory example of our paper for the the c.d.f of the standard
    normal distribution and the standard normal density for the gridpoint *c*
    for the real number line.'''

    convolution  = integrate.quad(lambda y: norm.cdf(c-y) * norm.pdf(y), -np.inf,np.inf)[0]
    return convolution

def convolution_cdf_squared_df(c):
    ''' Calculate the convolution as defined by Bühlmann and Yu (2002) and as
    used in the inductory example of our paper for the the squared c.d.f of the standard
    normal distribution and the standard normal density for the gridpoint *c*
    for the real number line.'''

    convolution  = integrate.quad(lambda y: norm.cdf(c-y) ** 2 * norm.pdf(y), -np.inf,np.inf)[0]
    return convolution

def calculate_bias_bagged(c_value):
    ''' Calculate the squared bias for the bagged predictor given the grid point
    *c_value*
    '''
    bias_bagged = (convolution_cdf_df(c_value) - norm.cdf(c_value)) ** 2
    return bias_bagged

def calculate_var_bagged(c_value):
    ''' Calculate the variance for the bagged predictor given the grid point
    *c_value*
    '''
    var_bagged = convolution_cdf_squared_df(c_value) - convolution_cdf_df(c_value) ** 2
    return var_bagged

def calculate_var_unbagged(c_value):
    ''' Calculate the variance for the bagged predictor given the grid point
    *c_value*
    '''
    var_unbagged = norm.cdf(c_value) * (1 - norm.cdf(c_value))
    return var_unbagged

def calculate_toy_example(settings):
    ''' TBT X-X
    '''

    # Create grid with *c* values that we want to consider.
    c_range = np.linspace(settings['c_min'],settings['c_max'], num=settings['c_gridpoints'])

    # Create the arrays that will be used to save the results.
    bagged_var = np.ones(settings['c_gridpoints']) * np.nan
    unbagged_var = np.ones(settings['c_gridpoints']) * np.nan
    bagged_bias = np.ones(settings['c_gridpoints']) * np.nan

    # We save all results to the dictonary *output*.
    output = {}
    output['c_range'] = c_range

    # Loop over *c* values that we want to consider and save the results.
    # Note that the unbagged predcitor is unbiased.
    for i_c, c in enumerate(c_range):
        bagged_var[i_c] = calculate_var_bagged(c)
        unbagged_var[i_c] = calculate_var_unbagged(c)
        bagged_bias[i_c] = calculate_bias_bagged(c)

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
    with open(ppj("IN_MODEL_SPECS","toy_example_settings.json")) as f:
        toy_example_settings_imported = json.load(f)

    calculate_toy_example = calculate_toy_example(toy_example_settings_imported)

    with open(ppj("OUT_ANALYSIS_THEORY","output_toy_example.pickle"), "wb") as out_file:
        pickle.dump(calculate_toy_example, out_file)
