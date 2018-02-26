"""
Created on Tue Dec 12 15:02:16 2017

@author: Tobias Werner

This module simulates the convergence of bagging/subagging towards a stable value

"""

import sys
from src.model_code.montecarlosimulation import MonteCarloSimulation
import numpy as np
import json
import pickle

from bld.project_paths import project_paths_join as ppj


def simulate_convergence(general_settings, convergence_settings, model):
    '''TBT X-X
    '''
    # MSE + Variance + Bias + Error = 4
    SIZE_MSE_DECOMP = 4

    # Create an array with all bootstrap itertions we consider.
    n_bootstraps_array = np.arange(
        convergence_settings['min_bootstrap'],
        convergence_settings['max_bootstrap'],
        convergence_settings['steps_bootstrap'])

    # Create an array to save the results.
    output_convergence = np.ones(
        (n_bootstraps_array.shape[0],
         SIZE_MSE_DECOMP)) * np.nan

    simulation_basis = MonteCarloSimulation(
        n_repeat=general_settings['n_repeat'],
        noise=general_settings['noise'],
        data_process=model,
        n_test_train=general_settings['n_test_train'],
        random_seeds=general_settings['random_seeds'])
    # Simulate over the range of bootstrap iteration values.
    for index, n_bootstrap in enumerate(n_bootstraps_array):
        output_convergence[index,
                           :] = simulation_basis.calc_mse(ratio=general_settings['BAGGING_RATIO'],
                                                          bootstrap=True,
                                                          min_split_tree=general_settings["min_split_tree"],
                                                          b_iterations=n_bootstrap)

    # Simulate MSE for a high number of bootstrap iterations to visualize its
    # convergence
    output_large_B = simulation_basis.calc_mse(
        ratio=general_settings['BAGGING_RATIO'],
        bootstrap=True,
        min_split_tree=general_settings["min_split_tree"],
        b_iterations=convergence_settings['converged_bootstrap'])
    return output_convergence, output_large_B


if __name__ == '__main__':
    dgp_model = sys.argv[1]

    with open(ppj("IN_MODEL_SPECS", "general_settings_small.json")) as f:
        general_settings_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS", "convergence_settings.json")) as f:
        convergence_settings_imported = json.load(f)

    output_simulation = simulate_convergence(
        general_settings_imported,
        convergence_settings_imported,
        dgp_model)

    simulation_convergence = {}
    simulation_convergence['bagging_range'] = output_simulation[0]
    simulation_convergence['bagging_large'] = output_simulation[1]

    with open(ppj("OUT_ANALYSIS_MAIN", "output_convergence_{}.pickle".format(dgp_model)), "wb") as out_file:
        pickle.dump(simulation_convergence, out_file)
    print('Done with the {} model for the convergence simulation'.format(dgp_model))
