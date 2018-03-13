"""
This module simulates the dependence of the subagging results on the
subsampling faction and sets it in relation to bagging.

For this we use the MonteCarloSimulation Class described in :ref:`model_code`
in the *simulate_bagging_subagging()* function and return the results as a
dictionary.

"""


import sys
import json
import pickle
from src.model_code.montecarlosimulation import MonteCarloSimulation

from bld.project_paths import project_paths_join as ppj


def simulate_bagging_subagging(general_settings, subagging_settings, model):
    """
    A  function that simulates the subsampling ratio dependency  of the Subagging
    Algorithm.

    Parameters
    ----------
    general_settings: Dictionary as described in :ref:`model_specs`
        The dictionary is shared across various simulations and defines the
        overall simulation set-up.

    subagging_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the
        subagging simulation.

    model: String that defines the data generating process to be considered.
        The option are 'friedman', 'linear' and 'indicator' which is usually
        passed as the first system argument.

    Returns a tuple of the simulation results:
        tuple[0]: numpy array of shape = 4
                  The array consists of the MSPE decompositions for the Bagging
                  Algorithm.
        tuple[1]: numpy array of shape = [*n_ratios*, 4], where *n_ratios* is
                  the number of subsampling ratios to be considered. This is
                  defined by keys in *subagging_settings*. The array consists
                  of the MSPE decompositions for each of those subsampling
                  fraction.

    """
    # Create a MonteCarloSimulation instance that defines the attributes For
    # the data generating process and will be constant for the subagging and
    # bagging simulation.
    simulation_basis = MonteCarloSimulation(
        n_repeat=general_settings['n_repeat'],
        noise=general_settings['noise'],
        data_process=model,
        n_test_train=general_settings['n_test_train'],
        random_seeds=general_settings['random_seeds'])

    # Perform the simulation for bagging.
    output_bagging = simulation_basis.calc_mse(
        ratio=general_settings['bagging_ratio'],
        bootstrap=True,
        min_split_tree=general_settings["min_split_tree"],
        b_iterations=general_settings["b_iterations"])

    # Peform the simulation for subagging for the given number of ratios.
    output_subagging = simulation_basis.calc_mse_all_ratios(
        n_ratios=subagging_settings["n_ratios"],
        min_ratio=subagging_settings["min_ratio"],
        max_ratio=subagging_settings["max_ratio"],
        min_split_tree=general_settings["min_split_tree"],
        b_iterations=general_settings["b_iterations"])

    return output_bagging, output_subagging


if __name__ == '__main__':
    DGP_MODEL = sys.argv[1]
    with open(ppj("IN_MODEL_SPECS", "general_settings.json")) as f:
        GENERAL_SETTINGS_IMPORTED = json.load(f)

    with open(ppj("IN_MODEL_SPECS", "subagging_settings.json")) as f:
        SUBAGGING_SETTINGS_IMPORTED = json.load(f)

    OUTPUT_SIMULATION = simulate_bagging_subagging(
        GENERAL_SETTINGS_IMPORTED, SUBAGGING_SETTINGS_IMPORTED, DGP_MODEL)

    SIMULATE_SUBAGGING = {}
    SIMULATE_SUBAGGING['bagging'] = OUTPUT_SIMULATION[0]
    SIMULATE_SUBAGGING['subagging'] = OUTPUT_SIMULATION[1]

    with open(ppj("OUT_ANALYSIS_MAIN", "output_subagging_{}.pickle"
                  .format(DGP_MODEL)), "wb") as out_file:
        pickle.dump(SIMULATE_SUBAGGING, out_file)
    print('Done with the Subagging Simulation for the {} model'.format(DGP_MODEL))
