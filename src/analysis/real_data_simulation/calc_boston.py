"""
This module simulates the MSPE for bagging and subagging for the Boston Housing data set. The simulation set-up is the
following:

i. For each simulation iteration follow this procedure
    (a) Randomly divide the data set into a training and a test set
    (b) Fit the predictor (Tree, Bagging, Subagging) to the training set
    (c) Using this new predictor make a prediction into the current test set and save the
        predicted values
    (d) Compute the average prediction error of the current test set and save the value
ii. Compute the MSPE as the mean of average prediction errors of each iteration

For this we use the BaggingTree Class described in :ref:`model_code` in the simulate_convergence() function
and return the results as a dictionary.

Parts of the simulation run in parallel using the joblib library.

"""


import sys
from src.model_code.baggingtree import BaggingTree
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
        b_iterations,
        min_split_tree,
        a_ratio):
    """
    A  function that splits the data consisting of *X* and *y* into a new test and training sample, fits the Bagging
    Algorithm on the training sample and makes a prediction on the test sample.
    Eventually the MSPE is computed.

    Parameters
    ----------
    X: numpy-array with shape = [n_size, n_features] (Default: None)
        The covariant matrix *X* with the sample size n_size and
        n_features of covariants.

    y: numpy-array with shape = [n_size] (Default: None)
        The vector of the dependent variable *y* with the sample size n_size

    ratio_test: float
        The ratio of the data used for the test sample.

    random_seed_split: int
        The random seed for the train_test_split.
        Note: It defines a RandomState instance and thus we don't reseed the module.

    random_seed_fit: int
        The random seed for the BaggingTree instance.
        Note: It defines a RandomState instance and thus we don't reseed the module.

    bootstrap_bool: bool
        Defines if we use the standard bootstrap or m out of n bootstrap.
        *bootstrap_bool=True* would imply that we use the standard bootstrap.

    b_iterations: int
        Number of Bootstrap iterations used to construct the Bagging Predictor.

    min_split_tree: int
        Defines the tree depth by setting the minimal size of a terminal node to be considered for a split.
    a_ratio: float
        Defines the subsampling ratio.

    Returns the MSPE for one iteration.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio_test, random_state=random_seed_split)
    bagging_instance = BaggingTree(
        random_seed=random_seed_fit,
        ratio=a_ratio,
        bootstrap=bootstrap_bool,
        b_iterations=b_iterations,
        min_split_tree=min_split_tree
    )
    fitted_model = bagging_instance.fit(X_train, y_train)
    predictions = fitted_model.predict(X_test)

    y_mse = np.mean((y_test - predictions) ** 2)
    return y_mse


def simulate_bagging_parallel(X, y, general_settings, boston_settings):
    """
    A  function that simulates the MSPE for the Bagging Algorithm using the Boston Housing data.

    Parameters
    ----------
    X: numpy-array with shape = [n_size, n_features] (Default: None)
        The covariant matrix *X* with the sample size n_size and
        n_features of covariants.

    y: numpy-array with shape = [n_size] (Default: None)
        The vector of the dependent variable *y* with the sample size n_size

    general_settings: Dictionary as described in :ref:`model_specs`
        The dictionary is shared across various simulations and defines the overall simulation set-up.

    boston_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the boston simulation.

    Returns the simulated MSPE.

    """

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
            b_iterations = general_settings['b_iterations'],
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
    """
    A  function that simulates the MSPE for the Subagging Algorithm over a range of subsampling fractions.

    Parameters
    ----------
    X: numpy-array with shape = [n_size, n_features] (Default: None)
        The covariant matrix *X* with the sample size n_size and
        n_features of covariants.

    y: numpy-array with shape = [n_size] (Default: None)
        The vector of the dependent variable *y* with the sample size n_size

    general_settings: Dictionary as described in :ref:`model_specs`
        The dictionary is shared across various simulations and defines the overall simulation set-up.

    subagging_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the subagging simulation.

    boston_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the boston simulation.

    Returns a numpy array with the simulated MSPE for each subsampling fraction.

    """

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
                b_iterations = general_settings['b_iterations'],
                min_split_tree = general_settings['min_split_tree'],
                a_ratio=a_value) for i in range(
                    general_settings['n_repeat']))
        mse_subagging[index_a] = np.mean(mse_not_expected)
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
