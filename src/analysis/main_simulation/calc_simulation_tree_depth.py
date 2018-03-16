"""
This module simulates the variations in the model complexity governed by the
Tree depth for the Bagging Algorithm.

For this we use the ``MonteCarloSimulation`` Class described in :ref:`model_code`
in the *simulate_tree_depth()* function and return the results as a dictionary.

"""
import sys
import json
import pickle
import numpy as np
from src.model_code.montecarlosimulation import MonteCarloSimulation

from bld.project_paths import project_paths_join as ppj


def simulate_tree_depth(general_settings, tree_depth_settings, model):
    """
    A  function that simulates the variations in tree depth an its effect on the
    MSPE decomposition for the Bagging Algorithm.

    Parameters
    ----------
    general_settings: Dictionary as described in :ref:`model_specs`
        The dictionary is shared across various simulations and defines the overall
        simulation set-up.

    tree_depth_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the tree
        depth simulation.

    model: String that defines the data generating process to be considered.
        The option are 'friedman', 'linear' and 'indicator' which is usually
        passed as the first system argument.

    Returns a tuple of the simulation results:
        tuple[0]: numpy array of shape = [min_split_array.size, 4], where
                  *min_split_array* is the array of minimal split values we want
                  to consider. This is defined by keys in *tree_depth_settings*.
                  The array consists of the MSPE decompositions for each of those
                  minimal split values for the Bagging Algorithm.
        tuple[0]: numpy array of shape = [min_split_array.size, 4], where
                  *min_split_array* is the array of minimal split values we want
                  to consider. This is defined by keys in *tree_depth_settings*.
                  The array consists of the MSPE decompositions for each of those
                  minimal split values for the **unbagged** Tree.
    """
    # MSE + Variance + Bias + Error = 4
    size_mse_decomp = 4
    # Create an array that describes minimal leaf sizes.
    # As we want to start from high to low, we turn the array around with
    # [::-1].
    # We add the step size again to the maximal value as we want it to be
    # included.
    min_split_array = (
        np.arange(
            tree_depth_settings['min_split'],
            tree_depth_settings['max_split'] +
            tree_depth_settings["steps_split"],
            tree_depth_settings["steps_split"]
        )[::-1]
    )

    # Create arrays to save the MSE, Bias, Variance + Noise for each split
    # specification.
    output_array_bagging = (
        np.ones((min_split_array.size, size_mse_decomp)) * np.nan
    )
    output_array_tree = (
        np.ones((min_split_array.size, size_mse_decomp)) * np.nan
    )

    # Create a MonteCarloSimulation instance that defines the attributes For
    # the data generating process and will be constant for the tree and
    # bagging simulation.
    simulation_basis = (
        MonteCarloSimulation(
            n_repeat=general_settings['n_repeat'],
            noise=general_settings['noise'],
            data_process=model,
            n_test_train=general_settings['n_test_train'],
            random_seeds=general_settings['random_seeds']
        )
    )
    # We simulate the MSE for Bagging and Trees for the different splits, while
    # keeping the data generating process constant.
    for index, split in enumerate(min_split_array):
        output_bagging = (
            simulation_basis.calc_mse(
                ratio=general_settings['bagging_ratio'],
                bootstrap=True,
                min_split_tree=split,
                b_iterations=general_settings["b_iterations"]
            )
        )
        # Note: Subagging(bootstrap=False) with ratio = 1 -> Tree
        output_tree = (
            simulation_basis.calc_mse(
                ratio=general_settings['bagging_ratio'],
                bootstrap=False,
                min_split_tree=split,
                b_iterations=general_settings["b_iterations"]
            )
        )

        output_array_bagging[index, :] = output_bagging
        output_array_tree[index, :] = output_tree

    return output_array_bagging, output_array_tree


if __name__ == '__main__':
    DGP_MODEL = sys.argv[1]

    with open(ppj("IN_MODEL_SPECS", "general_settings.json")) as f:
        GENERAL_SETTINGS_IMPORTED = json.load(f)

    with open(ppj("IN_MODEL_SPECS", "tree_depth_settings.json")) as f:
        TREE_DEPTH_SETTINGS_IMPORTED = json.load(f)

    OUTPUT_SIMULATION = simulate_tree_depth(
        GENERAL_SETTINGS_IMPORTED,
        TREE_DEPTH_SETTINGS_IMPORTED,
        DGP_MODEL)

    SIMULATION_TREE_DEPTH = {}
    SIMULATION_TREE_DEPTH['bagging'] = OUTPUT_SIMULATION[0]
    SIMULATION_TREE_DEPTH['trees'] = OUTPUT_SIMULATION[1]

    with open(ppj("OUT_ANALYSIS_MAIN", "output_tree_depth_{}.pickle"
                  .format(DGP_MODEL)), "wb") as out_file:
        pickle.dump(SIMULATION_TREE_DEPTH, out_file)
    print('Done with the {} model for the tree depth simulation'.format(DGP_MODEL))
