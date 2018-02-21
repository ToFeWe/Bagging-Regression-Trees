import sys
from src.model_code.simulations import MonteCarloSimulation
import numpy as np
import json
import pickle

from bld.project_paths import project_paths_join as ppj


def simulate_bagging_subagging(general_settings, subagging_settings, model):
    '''TBT X-X
    '''
    # Create a MonteCarloSimulation instance that defines the attributes For
    # the data generating process and will be constant for the subagging and
    # bagging simulation.
    simulation_basis = MonteCarloSimulation(n_repeat=general_settings['n_repeat'],
                                     noise = general_settings['noise'],
                                     data_process = model,
                                     n_test_train = general_settings['n_test_train'],
                                     random_seeds=general_settings['random_seeds'])

    # Perform the simulation for bagging.
    output_bagging = simulation_basis.calc_mse(ratio=general_settings['BAGGING_RATIO'], bootstrap=True, min_split_tree=general_settings["min_split_tree"], B_iterations = general_settings["B_iterations"])

    # Peform the simulation for subagging for the given number of ratios.
    output_subagging = simulation_basis.calc_mse_all_ratios(n_ratios=subagging_settings["n_ratios"], min_ratio=subagging_settings["min_ratio"],max_ratio=subagging_settings["max_ratio"], min_split_tree=general_settings["min_split_tree"], B_iterations = general_settings["B_iterations"])

    return output_bagging, output_subagging


if __name__ == '__main__':
    dgp_model = sys.argv[1]
    with open(ppj("IN_MODEL_SPECS","general_settings_small.json")) as f:
        general_settings_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS","subagging_settings.json")) as f:
        subagging_settings_imported = json.load(f)

    output_simulation = simulate_bagging_subagging(general_settings_imported, subagging_settings_imported, dgp_model)

    simulation_subagging = {}
    simulation_subagging['bagging'] = output_simulation[0]
    simulation_subagging['subagging'] = output_simulation[1]

    with open(ppj("OUT_ANALYSIS_MAIN","output_subagging_{}.pickle".format(dgp_model)), "wb") as out_file:
        pickle.dump(simulation_subagging, out_file)
    print('Done with the Subagging Simulation for the {} model'.format(dgp_model))
