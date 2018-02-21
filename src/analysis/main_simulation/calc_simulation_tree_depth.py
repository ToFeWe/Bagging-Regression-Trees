# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:49:40 2017

@author: Tobias Werner

This module calculates the simulated values for MSE, Bias, Variance for different
tree sizes. The tree size is regulated by specifing the minimal sample size a leaf
can have to be considered for a split.

"""

import sys
from src.model_code.simulations import MonteCarloSimulation
import numpy as np
import json
import pickle

from bld.project_paths import project_paths_join as ppj

def simulate_tree_depth(general_settings, tree_depth_settings, model):
    ''' TBT X-X
    '''
    # MSE + Variance + Bias + Error = 4
    SIZE_MSE_DECOMP = 4
    # Create an array that describes minimal leaf sizes.
    # As we want to start from high to low, we turn the array around with [::-1].
    min_split_array = np.arange(tree_depth_settings['min_split'],tree_depth_settings['max_split']+tree_depth_settings["steps_split"],tree_depth_settings["steps_split"])[::-1]

    # Create arrays to save the MSE, Bias, Variance + Noise for each split specification.
    output_array_bagging = np.ones((min_split_array.size,SIZE_MSE_DECOMP)) * np.nan
    output_array_tree = np.ones((min_split_array.size,SIZE_MSE_DECOMP)) * np.nan

    # Create a MonteCarloSimulation instance that defines the attributes For
    # the data generating process and will be constant for the tree and
    # bagging simulation.
    simulation_basis = MonteCarloSimulation(n_repeat=general_settings['n_repeat'],
                                     noise = general_settings['noise'],
                                     data_process = model,
                                     n_test_train = general_settings['n_test_train'],
                                     random_seeds=general_settings['random_seeds'])
    # We simulate the MSE for Bagging and Trees for the different splits, while
    # keeping the data generating process constant.
    for index, split in enumerate(min_split_array):
        output_bagging = simulation_basis.calc_mse(ratio=general_settings['BAGGING_RATIO'],
                                                   bootstrap=True,
                                                   min_split_tree=split,
                                                   B_iterations = general_settings["B_iterations"])
        # Note: Subagging(bootstrap=False) with ratio = 1 -> Tree
        output_tree = simulation_basis.calc_mse(ratio=general_settings['BAGGING_RATIO'],
                                                bootstrap=False,
                                                min_split_tree=split,
                                                B_iterations = general_settings["B_iterations"])



        output_array_bagging[index,:] = output_bagging
        output_array_tree[index,:] = output_tree
        print('Done with split:',split,' for the Model',model)

    return output_array_bagging, output_array_tree


if __name__ == '__main__':
    dgp_model = sys.argv[1]

    with open(ppj("IN_MODEL_SPECS","general_settings_small.json")) as f:
        general_settings_imported = json.load(f)


    with open(ppj("IN_MODEL_SPECS","tree_depth_settings.json")) as f:
        tree_depth_settings_imported = json.load(f)


    output_simulation = simulate_tree_depth(general_settings_imported, tree_depth_settings_imported, dgp_model)

    simulation_tree_depth = {}
    simulation_tree_depth['bagging'] = output_simulation[0]
    simulation_tree_depth['trees'] = output_simulation[1]

    with open(ppj("OUT_ANALYSIS_MAIN","output_tree_depth_{}.pickle".format(dgp_model)), "wb") as out_file:
        pickle.dump(simulation_tree_depth, out_file)
    print('Done with the {} model for the tree depth simulation'.format(dgp_model))
