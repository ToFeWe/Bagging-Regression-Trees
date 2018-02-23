# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:04:39 2017

@author: Tobias Werner
"""

import numpy as np
from scipy.stats import norm

import json
import pickle

from bld.project_paths import project_paths_join as ppj

def bias_normal_splits(c, a, gamma):
    ''' TBT X-X
    '''
    bias = (norm.cdf(c * a ** gamma) - norm.cdf(c)) ** 2
    return bias

def variance_normal_splits(c, a, gamma):
    ''' TBT X-X
    '''
    variance =  (a * norm.cdf(c * a ** gamma) * (1 - norm.cdf(c * a ** gamma)))
    return variance

def calculate_normal_splits(settings):
    '''TBT X-X
    '''
    output = {}

    # Create a range of c values that we will iterate over for each subsampling
    # fraction a and save it to output dictionary for plotting.
    c_range = np.linspace(settings['c_min'], settings['c_max'], num=settings['c_gridpoints'])
    output['c_range'] = c_range

    # Create the array with the a_range. The formation is chosen this way, for
    # plotting reasons.
    # Note that the first a is always one as a reference (unbagged). Hence no fraction.
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
    for i_a, a in np.ndenumerate(a_range):
        # For the list of a values (subsampling fraction).

        # Create an array that save the results for Bias and Variance
        bias_array = np.ones(settings['c_gridpoints']) * np.nan
        var_array = np.ones(settings['c_gridpoints']) * np.nan

        for i_c, c in np.ndenumerate(c_range):
            # The calculation are done straight forward following the derivations in the paper.
            bias_array[i_c] = bias_normal_splits(c, a , settings['gamma'])
            var_array[i_c] = variance_normal_splits(c, a, settings['gamma'])

        mse_array= np.add(bias_array, var_array)

        # Save the results to the dictonary. Note that we use the iteration number
        # as the key, since we follow a similar logic in the plotting part. As
        # *i_a* is a tuple with one entry we pick the first, hence [0].
        output[i_a[0]] = {}
        output[i_a[0]]['bias'] = bias_array
        output[i_a[0]]['variance'] = var_array
        output[i_a[0]]['mse'] = mse_array
    return output


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS","normal_splits_settings.json")) as f:
        normal_splits_settings_imported = json.load(f)

    calculate_normal_splits = calculate_normal_splits(normal_splits_settings_imported)

with open(ppj("OUT_ANALYSIS_THEORY","output_normal_splits.pickle"), "wb") as out_file:
    pickle.dump(calculate_normal_splits, out_file)
