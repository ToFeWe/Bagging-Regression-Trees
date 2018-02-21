# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:17:12 2017

@author: Tobias Werner

This module peforms the simulations needed.
"""

# Import the needed packages
import numpy as np
from src.model_code.draw_bootstrap import BaggingTree
from src.model_code.DGP import DataSimulation


class MonteCarloSimulation:
    """TBT
    Performs a Monte Carlo simulation for the given parameters

    """

    def __init__(self,
                 n_repeat=150,
                 noise=1,
                 data_process="friedman",
                 n_test_train=(500, 500),
                 random_seeds=(None, None, None, None),
                 ):
        self.n_repeat = n_repeat
        self.noise = noise
        self.data_process = data_process
        self.n_test = n_test_train[0]
        self.n_train = n_test_train[1]

        # Define the random states. For further details on why we do this, see
        # in the documentation.

        self.random_seed_noise = random_seeds[0]
        self.random_seed_fit = random_seeds[1]
        self.random_seed_train = random_seeds[2]
        self.random_seed_test = random_seeds[3]

        # Create one X_test and f_test that will be same for all following
        # simulation steps. This is important as we want to make for each
        # step a prediction on the same sample.
        self.test_simulation = DataSimulation(n_size=self.n_test,
                                             noise=self.noise,
                                             without_error=True,
                                             random_seed=self.random_seed_test)
        # Create test sample according to the given data generating process.
        if self.data_process == 'friedman':
            self.X_test, self.f_test = self.test_simulation.friedman_1_model()
        elif self.data_process == 'indicator':
            self.X_test, self.f_test = self.test_simulation.indicator_model()
        elif self.data_process == 'linear':
            self.X_test, self.f_test = self.test_simulation.linear_model()

    def calc_mse(self, ratio=1, bootstrap=True, min_split_tree=2, B_iterations=50):
        ''' Performs the simulation and returns MSE, Bias^2, Variance and Error for
        the specified simulation under bagging as a numpy array

        '''

        # Create numpy array that saves the MSPE for each training
        # observation.
        y_mse = np.ones(self.n_test) * np.nan

        # Create the instane of the bagging algorithm class, with the given
        # parameters, that will be used for the rest of the simulation run.
        bagging_instance = BaggingTree(random_seed=self.random_seed_fit,
                                        ratio=ratio, bootstrap=bootstrap,
                                        B_iterations=B_iterations,
                                        min_split_tree=min_split_tree)

        # To make results compareable and to get a smooth plot (we have
        # to limit *n_repeat* due to computation reasons), we create a
        # RandomState container for numpy to draw the noise terms. For further
        # information on why we do this, see in the documentation X-X.
        random_state_noise = np.random.RandomState(self.random_seed_noise)

        # We define the basis for drawing the training samples.
        # Note that we do that here as we want to draw the same sequence of
        # training samples for all subagging iterations.
        train_instance = DataSimulation(n_size=self.n_train,
                                        noise=self.noise,
                                        without_error=False,
                                        random_seed=self.random_seed_train)

        # Assign to the variable *draw_train* the according function of the
        # DataSimulation class/train_instance instance.
        # This is mainly for ease of execution and notation.
        if self.data_process == 'friedman':
            draw_train = train_instance.friedman_1_model
        elif self.data_process == 'indicator':
            draw_train = train_instance.indicator_model
        elif self.data_process == 'linear':
            draw_train = train_instance.linear_model

        # Create array to save prediction results and simulated y_test. Note that
        # we only save test samples as we also want to compute the noise.
        predictions = np.ones((self.n_test, self.n_repeat)) * np.nan
        simulated_y_test_all = np.ones((self.n_test, self.n_repeat)) * np.nan

        # Create an array to save the squared-error for all simulation runs
        y_se_all = np.zeros((self.n_repeat, self.n_test))

        # Peform the main simulation. Further explanation on this can be found
        # in the paper.
        for i in range(self.n_repeat):
            # Draw a new error term for the given f_test.
            y_test = self.f_test + \
                random_state_noise.normal(0, self.noise, self.n_test)
            # Draw a new training set.
            X_train, y_train = draw_train()

            # Save y_test for the simulation run.
            simulated_y_test_all[:, i] = y_test

            # Use the *bagging_instance* to estimate bagging given new training
            # sample.
            fitted_bagging = bagging_instance.fit(X_train, y_train)

            # Make a prediction on the test sample and save the squared-error.
            predictions[:, i] = fitted_bagging.predict(self.X_test)
            y_se_all[i,:] = (y_test - predictions[:, i]) ** 2

        # Compute the simulated expected squared-error, squared-bias,variance and noise
        # for each observation.
        y_mse = y_se_all.sum(axis=0) / self.n_repeat
        y_bias = (self.f_test - predictions.mean(axis=1)) ** 2
        y_var = np.var(predictions, axis=1)
        y_noise = np.var(simulated_y_test_all, axis=1)

        # Average over all test observation and save to results to numpy array.
        output = np.array([np.mean(y_mse),
                           np.mean(y_bias),
                           np.mean(y_var),
                           np.mean(y_noise)
                           ])

        return output

    def calc_mse_all_ratios(self, n_ratios=10, min_ratio=0.1, max_ratio=1, min_split_tree=2, B_iterations=50):
        ''' Returns the MSE, Bias^2, Variance and Error for
        the specified simulation under subagging as a numpy array for the range of
        ratios.
        - Add Dimesions!

        '''
        # Array must be of length four: MSPE, Bias, Variance, Noise
        ARRAY_LENGTH = 4
        # Create a range of subsampling ratios.
        ratiorange = np.linspace(min_ratio, max_ratio, n_ratios)

        # Create an array to save simulation for each ratio.
        output_array_subagging = np.ones((n_ratios, ARRAY_LENGTH)) * np.nan

        # We loop over all ratios and save the results to an array.
        for index, ratio in enumerate(ratiorange):
            output_array_subagging[index, :] = self.calc_mse(ratio=ratio,
                                                          bootstrap=False,
                                                          min_split_tree=min_split_tree,
                                                          B_iterations=B_iterations)
        return output_array_subagging
