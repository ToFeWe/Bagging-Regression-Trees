# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:53:11 2017

@author: Tobias Werner

The module implements different data generating processes within
the *DataSimulation* class. In order to make results compareable
we define the attributes of the DGP within a class. All function then have the same
noise, size and random_seed.
"""

import numpy as np


class DataSimulation:
    """
    A  class that collect different data generating processes.

    Parameters
    ----------
    random_seed: int or None, optional (Default: None)
        random_seed is used to specify the RandomState for numpy.random.
        It is shared accros all functions.

        IMPORTANT: This random seed is fixed for a specific instance, as it specifies a new RandomState for all numpy functions
        used in the class. As a result this random_seed is *not* overwritten by numpy
        random seeds that are defined outside of specific class instance. The reason For
        this is that it makes reproducibility easier accross different simulations and
        modules.
        Note however that the downside is, that we have to specify for each class (each instance)
        a different random seed and it is not possible to specify one random seed at the begiing
        of the whole simulation, as this will define the RandomState within each class.

    n_size: int, optional (Default=500)
        The sample size, when calling one of the data geenrating functions.

    noise: int, float, optional (Default=0.0)
        The variance of the error term that is used for the data geenrating
        processes.
        The default of *noise* = 1.0 indicates that we draw without an error term
        that is standard normally distributed.

    without_error: bool, optional(Default=False)
        Specify if the data should be generated with an error term already added.

        Use this option if you want to create a test sample for which you draw
        an error term for each simulation iteration.
    """

    def __init__(
            self,
            random_seed=None,
            n_size=500,
            noise=1.0,
            without_error=False):
        self._check_random_seed(random_seed)
        self._check_n_size(n_size)
        self._check_noise(noise)
        self._check_without_error(without_error)

    def _check_random_seed(self, random_seed):
        if random_seed is None or isinstance(random_seed, int):
            self.random_seed = random_seed
            self.random_state = np.random.RandomState(random_seed)
        else:
            raise ValueError(
            'Must pass integer value or None as the random seed for the data generating process.'
            )

    def _check_n_size(self, n_size):
        if not isinstance(n_size, int) or n_size<1:
            raise ValueError(
            'Must pass a postitive integer as the sample size for the process'
            )
        else:
            self.n_size = n_size

    def _check_noise(self, noise):
        if isinstance(noise, (float, int)):
            self.noise = noise
        else:
            raise ValueError(
            'Must pass float or int as the noise term.'
            )
    def _check_without_error(self, without_error):
        if isinstance(without_error, bool):
            self.without_error = without_error
        else:
            raise ValueError(
            'Must pass bool as without_error to specify if you want'+
            'to draw with or without error term.'
            )


    def friedman_1_model(self):
        """
        Returns the Friedman #1 Model covariante matrix *X* and the
        target variable *y* as numpy arrays.
        Note that x6 to x10 do not contribute to y and can be considered as
        'noise' variables.

        For further reference see:
        Friedman, Jerome H. "Multivariate adaptive regression splines." The annals of statistics (1991): 1-67.
        """
        X = self.random_state.uniform(low=0, high=1, size=(self.n_size, 10))
        f_x = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1]) +
            20 * np.power((X[:, 2] - 0.5), 2) +
            10 * X[:, 3] + 5 * X[:, 4]
        )
        # Indicate if we draw an error term directly
        # and draw error if desired
        if self.without_error:
            y = f_x
        else:
            draw_e = self.random_state.normal(0, self.noise, self.n_size)
            y = np.add(draw_e, f_x)


        return X, y


    def linear_model(self):
        ''' Returns the linear model from Friedman Hall (2000) covariante matrix *X* and the
        target variable *y* as numpy arrays.

        For further reference see:

        Friedman, Jerome H., and Peter Hall. "On bagging and nonlinear estimation." Journal of statistical planning and inference 137.3 (2007): 669-683.
        '''

        X = self.random_state.uniform(low=0, high=1, size=(self.n_size, 10))

        f_x = (1 * X[:, 0] + 2 * X[:, 1] + 3 *
               X[:, 2] + 4 * X[:, 3] + 5 * X[:, 4])


        # Indicate if we draw an error term directly
        # and draw error if desired
        if self.without_error:
            y = f_x
        else:
            draw_e = self.random_state.normal(0, self.noise, self.n_size)
            y = np.add(draw_e, f_x)
        return X, y


    def _indicator_function(self, var_1, arg_1, var_2=None, arg_2=None):
        ''' Returns an numpy array with {0,1}

        '''
        equal_1 = np.equal(
            np.full(
                (1,
                 self.n_size)[0],
                arg_1,
                dtype='int32'),
            var_1)
        if var_2 is None and arg_2 is None:
            return 1 * equal_1
        else:
            equal_2 = np.equal(
                np.full(
                    (1, self.n_size)[0], arg_2, dtype='int32'), var_2)
            return 1 * np.logical_and(equal_1, equal_2)

    def indicator_model(self):
        '''Returns the Bühlman M3 Model as a numpy array
        '''
        # Initalize the variables covariante matrix
        # Note we always add +1 to the desired variable due to the
        # properties of numpy arrays (last value excluded)
        x_1 = self.random_state.randint(0, 1 + 1, size=(self.n_size))
        x_2 = self.random_state.randint(0, 1 + 1, size=(self.n_size))
        x_3 = self.random_state.randint(0, 3 + 1, size=(self.n_size))
        x_4 = self.random_state.randint(0, 3 + 1, size=(self.n_size))
        x_5 = self.random_state.randint(0, 7 + 1, size=(self.n_size))

        # Initalize the regression function of the indicator model
        f_x = (1 *
               self._indicator_function(x_1, 0) -
               3 *
               self._indicator_function(x_1, 1) +
               0.5 *
               self._indicator_function(x_2, 1) +
               2 *
               self._indicator_function(x_2, 1) +
               0.8 *
               self._indicator_function(x_3, 0) -
               2 *
               self._indicator_function(x_3, 1) +
               2 *
               self._indicator_function(x_3, 2) -
               1 *
               self._indicator_function(x_3, 3) +
               0.5 *
               self._indicator_function(x_4, 0) +
               1.2 *
               self._indicator_function(x_4, 1) -
               0.9 *
               self._indicator_function(x_4, 2) +
               1.8 *
               self._indicator_function(x_4, 3) +
               0.3 *
               self._indicator_function(x_5, 0) -
               0.6 *
               self._indicator_function(x_5, 1) +
               0.9 *
               self._indicator_function(x_5, 2) -
               1.2 *
               self._indicator_function(x_5, 3) +
               1.5 *
               self._indicator_function(x_5, 4) -
               1.8 *
               self._indicator_function(x_5, 5) +
               2.1 *
               self._indicator_function(x_5, 6) -
               2.4 *
               self._indicator_function(x_5, 7) +
               2 *
               self._indicator_function(x_1, 0, x_2, 1) +
               3 *
               self._indicator_function(x_2, 0, x_3, 1))
        # Create the covariante matrix by stacking the corresponding variables
        X = np.stack((x_1, x_2, x_3, x_4, x_5), axis=1)

        # Indicate if we draw an error term directly
        # and draw error if desired
        if self.without_error:
            y = f_x
        else:
            draw_e = self.random_state.normal(0, self.noise, self.n_size)
            y = np.add(draw_e, f_x)
        return X, y
