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

"""


from src.model_code.baggingtree import BaggingTree
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pickle

from bld.project_paths import project_paths_join as ppj
import json


def split_fit_predict_bagging(
        X,
        y,
        ratio_test,
        random_state,
        bagging_object):
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

    random_state: Numpy RandomState container
        The RandomState instance to perform the train/test split.

    bagging_object: Instance of the class *BaggingTrees*
        The bagging instance used to fit the algorithm to the newly splitted
        data.

    Returns the MSPE for one iteration.
    """
    # Split the sample into train and test using the RandomState instance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio_test, random_state=random_state)

    # Fit the Model and make a prediction on the test sample
    fitted_model = bagging_object.fit(X_train, y_train)
    predictions = fitted_model.predict(X_test)

    # Calculate the MSPE
    y_mse = np.mean((y_test - predictions) ** 2)
    return y_mse


def simulate_bagging(X, y, general_settings, boston_settings):
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
    # Define a RandomState for the train_test_split.
    random_state_split = np.random.RandomState(
        boston_settings['random_seed_split'])

    # Train the bagged Regression Tree.
    # The random seed can be fixed here.
    bagging_instance = BaggingTree(
        random_seed=boston_settings['random_seed_fit'],
        ratio=general_settings['bagging_ratio'],
        bootstrap=True,
        b_iterations=general_settings['b_iterations'],
        min_split_tree=general_settings['min_split_tree']
    )
    mse_not_expected = np.ones(general_settings['n_repeat']) * np.nan

    for i_n_repeat in range(general_settings['n_repeat']):
        mse_not_expected[i_n_repeat] = split_fit_predict_bagging(
            X,
            y,
            ratio_test=boston_settings['ratio_test'],
            random_state=random_state_split,
            bagging_object=bagging_instance
        )

    mse_sim = np.mean(mse_not_expected)
    return mse_sim


def simulate_subagging(
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
    ratio_range = np.linspace(
        subagging_settings['min_ratio'],
        subagging_settings['max_ratio'],
        subagging_settings['n_ratios'])

    for i_a, a_value in enumerate(ratio_range):
        # Define a RandomState for the train_test_split
        # Note:Inside the loop as we want to have a smooth plot -> same
        # samples.
        random_state_split = np.random.RandomState(
            boston_settings['random_seed_split'])

        # Train the bagged Regression Tree.
        # The random seed can be fixed here.
        bagging_instance = BaggingTree(
            random_seed=boston_settings['random_seed_fit'],
            ratio=a_value,
            bootstrap=False,
            b_iterations=general_settings['b_iterations'],
            min_split_tree=general_settings['min_split_tree']
        )
        mse_not_expected = np.ones(general_settings['n_repeat']) * np.nan

        for i_n_repeat in range(general_settings['n_repeat']):
            mse_not_expected[i_n_repeat] = split_fit_predict_bagging(
                X,
                y,
                ratio_test=boston_settings['ratio_test'],
                random_state=random_state_split,
                bagging_object=bagging_instance
            )

        mse_subagging[i_a] = np.mean(mse_not_expected)
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

    mse_bagging = simulate_bagging(
        boston_X,
        boston_y,
        general_settings_imported,
        boston_settings_imported
    )
    mse_subagging = simulate_subagging(
        boston_X,
        boston_y,
        general_settings_imported,
        subagging_settings_imported,
        boston_settings_imported
    )
    print(mse_bagging)
    print(mse_subagging)
    simulation_boston = {}
    simulation_boston['mse_bagging'] = mse_bagging
    simulation_boston['mse_subagging'] = mse_subagging

    with open(ppj("OUT_ANALYSIS_REAL_DATA", "output_boston.pickle"), "wb") as out_file:
        pickle.dump(simulation_boston, out_file)
    print('Done with the real data simulation')
