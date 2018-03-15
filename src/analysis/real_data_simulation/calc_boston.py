"""
This module simulates the MSPE for bagging and subagging for the Boston Housing
data set. The simulation set-up is the following:

i. For each simulation iteration follow this procedure
    (a) Randomly divide the data set into a training and a test set
    (b) Fit the predictor (Tree, Bagging, Subagging) to the training set
    (c) Using this new predictor make a prediction into the current test set
        and save the predicted values
    (d) Compute the average prediction error of the current test set and save
        the value
ii. Compute the MSPE as the mean of average prediction errors of each iteration

For this we use the BaggingTree Class described in :ref:`model_code` in the
*simulate_bagging()* and *simulate_subagging()* functions and write the
results as a dictionary.

"""

import pickle
import json
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from src.model_code.baggingtree import BaggingTree
from bld.project_paths import project_paths_join as ppj


def split_fit_predict_bagging(
        x_matrix,
        y_vector,
        ratio_test,
        random_state,
        bagging_object):
    """
    A  function that splits the data consisting of *x_matrix* and *y_vector*
    into a new test and training sample, fits the Bagging Algorithm on the
    training sample and makes a prediction on the test sample.
    Eventually the MSPE is computed.

    Parameters
    ----------
    x_matrix: numpy-array with shape = [n_size, n_features] (Default: None)
        The covariant matrix *x_matrix* with the sample size n_size and
        n_features of covariants.

    y_vector: numpy-array with shape = [n_size] (Default: None)
        The vector of the dependent variable *y_vector* with the sample size
        n_size.

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
    x_matrix_train, x_matrix_test, y_vector_train, y_vector_test = (
        train_test_split(
            x_matrix,
            y_vector,
            test_size=ratio_test,
            random_state=random_state
        )
    )

    # Fit the Model and make a prediction on the test sample.
    fitted_model = bagging_object.fit(x_matrix_train, y_vector_train)
    predictions = fitted_model.predict(x_matrix_test)

    # Calculate the MSPE.
    y_mse = np.mean((y_vector_test - predictions) ** 2)
    return y_mse


def simulate_bagging(x_matrix, y_vector, general_settings, boston_settings):
    """
    A  function that simulates the MSPE for the Bagging Algorithm using the
    Boston Housing data.

    Parameters
    ----------
    x_matrix: numpy-array with shape = [n_size, n_features] (Default: None)
        The covariant matrix *x_matrix* with the sample size n_size and
        n_features of covariants.

    y_vector: numpy-array with shape = [n_size] (Default: None)
        The vector of the dependent variable *y_vector* with the sample size
        n_size

    general_settings: Dictionary as described in :ref:`model_specs`
        The dictionary is shared across various simulations and defines the
        overall simulation set-up.

    boston_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the
        boston simulation.

    Returns the simulated MSPE.

    """
    # Define a RandomState for the train_test_split.
    random_state_split = (
        np.random.RandomState(boston_settings['random_seed_split'])
    )

    # Create an array that will save the simulation results.
    mse_not_expected = np.ones(general_settings['n_repeat']) * np.nan

    # Train the bagged Regression Tree.
    # The random seed can be fixed here.
    bagging_instance = (
        BaggingTree(
            random_seed=boston_settings['random_seed_fit'],
            ratio=general_settings['bagging_ratio'],
            bootstrap=True,
            b_iterations=general_settings['b_iterations'],
            min_split_tree=general_settings['min_split_tree']
        )
    )

    for i_n_repeat in range(general_settings['n_repeat']):
        mse_not_expected[i_n_repeat] = (
            split_fit_predict_bagging(
                x_matrix,
                y_vector,
                ratio_test=boston_settings['ratio_test'],
                random_state=random_state_split,
                bagging_object=bagging_instance
            )
        )

    # Average over all simulation results to get the MSPE.
    mse_sim = np.mean(mse_not_expected)
    return mse_sim


def simulate_subagging(
        x_matrix,
        y_vector,
        general_settings,
        subagging_settings,
        boston_settings):
    """
    A  function that simulates the MSPE for the Subagging Algorithm over a
    range of subsampling fractions.

    Parameters
    ----------
    x_matrix: numpy-array with shape = [n_size, n_features] (Default: None)
        The covariant matrix *x_matrix* with the sample size n_size and
        n_features of covariants.

    y_vector: numpy-array with shape = [n_size] (Default: None)
        The vector of the dependent variable *y_vector* with the sample size
        n_size

    general_settings: Dictionary as described in :ref:`model_specs`
        The dictionary is shared across various simulations and defines the
        overall simulation set-up.

    subagging_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to
        subagging simulations.

    boston_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the
        boston simulation.

    Returns a numpy array with the simulated MSPE for each subsampling fraction.

    """
    # Create an array that will save the MSPE for all ratios.
    mse_subagging = np.ones(subagging_settings['n_ratios']) * np.nan

    # Create the range of ratios we want to consider.
    ratio_range = (
        np.linspace(
            subagging_settings['min_ratio'],
            subagging_settings['max_ratio'],
            subagging_settings['n_ratios']
        )
    )

    # Loop over the different ratios we consider and simulate the MSPE for each ratio.
    for i_a, a_value in enumerate(ratio_range):
        # Define a RandomState for the train_test_split
        # Note:Inside the loop as we want to have a smooth plot -> same
        # samples.
        random_state_split = (
            np.random.RandomState(
                boston_settings['random_seed_split']
            )
        )

        # Train the bagged Regression Tree.
        # The random seed can be fixed here.
        bagging_instance = (
            BaggingTree(
                random_seed=boston_settings['random_seed_fit'],
                ratio=a_value,
                bootstrap=False,
                b_iterations=general_settings['b_iterations'],
                min_split_tree=general_settings['min_split_tree']
            )
        )
        # Create an array that will save the simulation results for one ratio.
        mse_not_expected = np.ones(general_settings['n_repeat']) * np.nan

        for i_n_repeat in range(general_settings['n_repeat']):
            mse_not_expected[i_n_repeat] = (
                split_fit_predict_bagging(
                    x_matrix,
                    y_vector,
                    ratio_test=boston_settings['ratio_test'],
                    random_state=random_state_split,
                    bagging_object=bagging_instance
                )
            )
        # Average over all simulation results for a specific ratio to get the MSPE.
        mse_subagging[i_a] = np.mean(mse_not_expected)
    return mse_subagging


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS", "general_settings.json")) as f:
        GENERAL_SETTINGS_IMPORTED = json.load(f)

    with open(ppj("IN_MODEL_SPECS", "boston_settings.json")) as f:
        BOSTON_SETTINGS_IMPORTED = json.load(f)

    with open(ppj("IN_MODEL_SPECS", "subagging_settings.json")) as f:
        SUBAGGING_SETTINGS_IMPORTED = json.load(f)

    BOSTON_FULL = load_boston()
    BOSTON_X_MATRIX = BOSTON_FULL['data']
    BOSTON_Y_VECTOR = BOSTON_FULL['target']

    MSE_BAGGING = (
        simulate_bagging(
            BOSTON_X_MATRIX,
            BOSTON_Y_VECTOR,
            GENERAL_SETTINGS_IMPORTED,
            BOSTON_SETTINGS_IMPORTED
        )
    )
    MSE_SUBAGGING = (
        simulate_subagging(
            BOSTON_X_MATRIX,
            BOSTON_Y_VECTOR,
            GENERAL_SETTINGS_IMPORTED,
            SUBAGGING_SETTINGS_IMPORTED,
            BOSTON_SETTINGS_IMPORTED
        )
    )

    SIMULATION_BOSTON = {}
    SIMULATION_BOSTON['mse_bagging'] = MSE_BAGGING
    SIMULATION_BOSTON['mse_subagging'] = MSE_SUBAGGING

    with open(ppj("OUT_ANALYSIS_REAL_DATA", "output_boston.pickle"), "wb") as out_file:
        pickle.dump(SIMULATION_BOSTON, out_file)
    print('Done with the real data simulation')
