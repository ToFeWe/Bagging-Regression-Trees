# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:07:27 2018

@author: Tobias Werner
"""

import sys
from src.model_code.bagging_module import BaggingTree
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pickle

from bld.project_paths import project_paths_join as ppj
import json

try:
    from joblib import Parallel, delayed
except ImportError:
    sys.exit("""You need joblib to run the simulations, which is installed by
                the enviroment setup provided with this project.
                Else install it from https://pythonhosted.org/joblib/
                or run:> pip install joblib.""")


def split_fit_predict_bagging(
        X,
        y,
        ratio_test,
        random_seed_split,
        random_seed_fit,
        bootstrap_bool,
        B_iterations,
        min_split_tree,
        a_ratio):
    '''
    Split into test and training set. Note that the Random Seed is changed
    for each iteration but still deterministic overall

    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio_test, random_state=random_seed_split)
    bagging_instance = BaggingTree(
        random_seed=random_seed_fit,
        ratio=a_ratio,
        bootstrap=bootstrap_bool,
        B_iterations=B_iterations,
        min_split_tree=min_split_tree
    )
    fitted_model = bagging_instance.fit(X_train, y_train)
    predictions = fitted_model.predict(X_test)

    y_mse = np.mean((y_test - predictions) ** 2)
    return y_mse


def simulate_bagging_parallel(X, y, general_settings, boston_settings):
    ''' Runs the simulation for bagging in parallel.
    '''
    # Note that you can change n_jobs to 1 in case your system suffers too much

    # Calculate the mse for each iteration in parallel and compute the MSE.
    # Note that we change the random seed for each iteration of Parallel.
    # Surely, this is not best practice, but it works well and safe in this
    # context.
    mse_not_expected = Parallel(
        n_jobs=boston_settings['cores'])(
        delayed(split_fit_predict_bagging)(
            X,
            y,
            ratio_test=boston_settings['ratio_test'],
            random_seed_split=boston_settings['random_seed_split'] + i,
            random_seed_fit=boston_settings['random_seed_fit'],
            bootstrap_bool=True,
            B_iterations = general_settings['B_iterations'],
            min_split_tree = general_settings['min_split_tree'],
            a_ratio=general_settings['BAGGING_RATIO']) for i in range(
                general_settings['n_repeat']))
    mse_sim = np.mean(mse_not_expected)
    return mse_sim


def simulate_subagging_parallel(
        X,
        y,
        general_settings,
        subagging_settings,
        boston_settings):
    ''' Runs the simulation for subagging in parallel.
    '''

    mse_subagging = np.ones(subagging_settings['n_ratios']) * np.nan
    ratiorange = np.linspace(subagging_settings['min_ratio'], subagging_settings['max_ratio'], subagging_settings['n_ratios'])
    for index_a, a_value in enumerate(ratiorange):
        # Calculate the mse for each iteration in parallel and compute the MSE.
        # Note that we change the random seed for each iteration of Parallel.
        # Surely, this is not best practice, but it works well and safe in this
        # context.
        mse_not_expected =  Parallel(n_jobs=boston_settings['cores'])(delayed(split_fit_predict_bagging)(
                X,
                y,
                ratio_test=boston_settings['ratio_test'],
                random_seed_split=boston_settings['random_seed_split'] + i,
                random_seed_fit=boston_settings['random_seed_fit'],
                bootstrap_bool=False,
                B_iterations = general_settings['B_iterations'],
                min_split_tree = general_settings['min_split_tree'],
                a_ratio=a_value) for i in range(
                    general_settings['n_repeat']))
        mse_subagging[index_a] = np.mean(mse_not_expected)
        print(np.mean(mse_not_expected), a_value)
    return mse_subagging


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS", "general_settings.json")) as f:
        general_settings_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS", "boston_settings.json")) as f:
        boston_settings_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS", "subagging_settings.json")) as f:
        subagging_settings_imported = json.load(f)

    boston_full = load_boston()
    boston_X = boston_full['data']
    boston_y = boston_full['target']

    mse_bagging = simulate_bagging_parallel(boston_X, boston_y, general_settings_imported, boston_settings_imported)
    mse_subagging = simulate_subagging_parallel(
        boston_X, boston_y, general_settings_imported, subagging_settings_imported, boston_settings_imported)
    print('MSE for Bagging: ', mse_bagging)
    print('MSE for the different subsampling ratios: \n', mse_subagging)
    simulation_boston = {}
    simulation_boston['mse_bagging'] = mse_bagging
    simulation_boston['mse_subagging'] = mse_subagging

    with open(ppj("OUT_ANALYSIS_REAL_DATA","output_boston.pickle"), "wb") as out_file:
        pickle.dump(simulation_boston, out_file)
