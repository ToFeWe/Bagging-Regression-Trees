# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:08:09 2017

@author: gl1rk
"""
import numpy as np
import json
import pickle

from bld.project_paths import project_paths_join as ppj

def indicator(x_value,Y_bar):
    ''' TBT X-X
    '''
    if Y_bar <= x_value:
        return 1
    if Y_bar > x_value:
        return 0


def bagged_indicator(x_value, sample, B_iterations=50):
    ''' TBT X-X
    '''
    predictions = np.ones(B_iterations) * np.nan
    for m in range(B_iterations):
        bootstrapsample = np.random.choice(sample,size=(sample.size,),replace=True)
        Y_bootstrap = bootstrapsample.mean()
        predictions[m] = indicator(x_value,Y_bootstrap)
    return predictions.mean()

def simulate_finite_sample(settings):
    ''' TBT X-X
    '''
    # Create dict to save the finale results.
    output = {}

    # Create array with x values we want to consider.
    x_range = np.linspace(settings['x_min'],
                            settings['x_max'],
                            settings['x_gridpoints'])

    # Save x_range to Dictionary as we want to plot the results later.
    output['x_range'] = x_range

    # Iterate over the list of sample sizes.
    for sample_size in settings['n_list']:
        # Create Arrays to save the results for the given sample size.
        mse_array_bagging = np.ones(settings['x_gridpoints']) * np.nan
        mse_array_unbagged = np.ones(settings['x_gridpoints']) * np.nan

        # Iterate over the range of x values.
        for i_x, x in enumerate(x_range):

            # Create Arrays to save the simulated results.
            y_se_bagged = np.ones(settings['n_repeat']) * np.nan
            y_se_unbagged = np.ones(settings['n_repeat']) * np.nan

            # Set random state s.t. for each grid point we draw the same sequence
            random_state = np.random.RandomState(settings['random_seed'])
            # Calculate the true prediction for the given x.
            true_prediction = indicator(x,settings['mu'])

            # Simulate the Expected MSPE for given x.
            for i_repeat in range(settings['n_repeat']):

                # Draw a new sample and make a prediction for bagging and without bagging
                Y_sample = random_state.normal(settings['mu'], settings['sigma'], size=sample_size)
                prediction_unbagged = indicator(x, Y_sample.mean())
                prediction_bagged = bagged_indicator(x, Y_sample, B_iterations=settings['B_iterations'])

                # Calculate the Squared Error for the given repetition
                y_se_bagged[i_repeat] = (true_prediction - prediction_bagged) ** 2
                y_se_unbagged[i_repeat] = (true_prediction - prediction_unbagged) ** 2

            # Calculate the MSPE
            mse_array_bagging[i_x] = y_se_bagged.sum(axis=0) / settings['n_repeat']
            mse_array_unbagged[i_x] = y_se_unbagged.sum(axis=0) / settings['n_repeat']

        # Save the results of the given sample size.
        output[sample_size] = {}
        output[sample_size]['mse_bagging'] = mse_array_bagging
        output[sample_size]['mse_unbagged'] = mse_array_unbagged

    return output




if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS","finite_sample_settings.json")) as f:
        finite_sample_settings_imported = json.load(f)

    simulation_finite_sample = simulate_finite_sample(finite_sample_settings_imported)

    with open(ppj("OUT_ANALYSIS_THEORY","output_finite_sample.pickle"), "wb") as out_file:
        pickle.dump(simulation_finite_sample, out_file)
