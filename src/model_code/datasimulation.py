"""

This module implements different data generating processes within
the *DataSimulation* class. In order to make the results for different functional
forms of f(x_matrix) comparable, we define the attributes of the datasimulation
within a class. All function to which we apply the Bagging Algorithm then have
the same noise, size and random_seed This is important, as we want to compare
its performance on different regression functions.

"""

import numpy as np


class DataSimulation:
    """
    A  class that collects different data generating processes.

    Parameters
    ----------
    random_seed: int or None, optional (Default: None)
        random_seed is used to specify the RandomState for numpy.random.
        It is shared across all functions of the class.

        Needs to be specified as a usual random seed
        as it is deployed to numpy.random.

        IMPORTANT: This random seed is fixed for a specific instance, as it
        specifies a new RandomState for all numpy functions used in this class.
        As a result this random_seed is *not* overwritten by numpy random seeds
        that are defined outside of specific class instance. The reason for
        this is that it makes reproducibility easier across different simulations
        and modules. Note however that the downside is, that we have to specify
        for each class (each instance) a different random seed and it is not
        possible to specify one random seed at the beginning of the whole
        simulation, as this will define the RandomState within each class.

        For further information on this see in :ref:`design_choices`.

    n_size: int, optional (Default=500)
        The sample size, when calling one of the data generating functions.

        Needs to be greater than 0.

    noise: int, float, optional (Default=1.0)
        The variance of the error term that is used for the data generating
        processes.
        The default of *noise* = 1.0 indicates that we draw an error term
        that is standard normally distributed.

    without_error: bool, optional(Default=False)
        Specify if the data should be generated with an error term already added.

        Default=False implies that it is created *with* an error term.
        Change this option to *True* if you want to create a test sample for
        which you draw an error term for each simulation iteration.

    """

    def __init__(
            self,
            random_seed=None,
            n_size=500,
            noise=1.0,
            without_error=False):

        # We define the random state. For further details on why we do this, see
        # in the documentation.
        # No function to check the seed as it is done directly by numpy.
        self.random_state = np.random.RandomState(random_seed)

        # Check and then set the inputs to avoid programming errors.
        self._set_n_size(n_size)
        self._set_noise(noise)
        self._set_without_error(without_error)

    def _set_n_size(self, n_size):
        """ A function to check if *n_size* is specified correctly."""
        assert (np.issubdtype(type(n_size), np.integer) and n_size > 0), \
            ('*n_size* need to be an integer greater than zero.'
             ' You provided b_iteartions={}, which is of type {}.'
             ''.format(n_size, type(n_size)))
        self.n_size = n_size

    def _set_noise(self, noise):
        """ A function to check if *noise* is specified correctly. """
        assert (np.issubdtype(type(noise), np.float) and noise > 0), \
            ('*noise* needs to be of type integer or float and greater or equal'
             ' to zero. You provided noise={}, which is of type {}.'
             ''.format(noise, type(noise)))
        self.noise = noise

    def _set_without_error(self, without_error):
        """ A function to check if *without_error* is specified correctly. """
        assert isinstance(without_error, bool), \
            ('*without_error* needs to be of type *bool*. '
             'The provided value is of type {}'.format(type(without_error)))
        self.without_error = without_error

    def _set_y(self, f_vector):
        """
        Adds the error term to *f_vector* if we draw with error.
        If we draw without error, it returns *f_vector* again.
        """
        if self.without_error:
            y_vector = f_vector
        else:
            error_vector = self.random_state.normal(0, self.noise, self.n_size)
            y_vector = np.add(error_vector, f_vector)
        return y_vector

    def friedman_1_model(self):
        """
        Returns the Friedman #1 Model by :cite:`friedman1991` covariant matrix
        *x_matrix* (shape = [n_size, 10]) and the target variable *y_vector*
        (shape = [n_size])as a numpy arrays for the values specified in the
        class instance. Note that x6 to x10 do not contribute to y_vector and can be
        considered as 'noise' variables.

        For the full functional form is given in the paper and
        :cite:`friedman1991`.
        """
        x_matrix = (
            self.random_state.uniform(low=0, high=1, size=(self.n_size, 10))
        )
        f_x = (
            10 * np.sin(np.pi * x_matrix[:, 0] * x_matrix[:, 1]) +
            20 * np.power((x_matrix[:, 2] - 0.5), 2) +
            10 * x_matrix[:, 3] + 5 * x_matrix[:, 4]
        )
        # Add an error term to *f_x* if this is desired.
        y_vector = self._set_y(f_x)

        return x_matrix, y_vector

    def linear_model(self):
        """ Returns the linear model from :cite:`friedman2007` covariant matrix
        *x_matrix* (shape = [n_size, 10]) and the target variable *y_vector*
        (shape = [n_size]) as numpy arrays for the values specified in the
        class instance. Note that x6 to x10 do not contribute to y_vector and can be
        considered as 'noise' variables.

        For the full functional form is given in the paper and
        :cite:`friedman2007`.
        """

        x_matrix = (
            self.random_state.uniform(low=0, high=1, size=(self.n_size, 10))
        )

        f_x = (
            1 * x_matrix[:, 0] + 2 * x_matrix[:, 1] + 3 *
            x_matrix[:, 2] + 4 * x_matrix[:, 3] + 5 * x_matrix[:, 4]
        )

        # Add an error term to *f_x* if this is desired.
        y_vector = self._set_y(f_x)

        return x_matrix, y_vector

    def _indicator_function(self, var_1, arg_1, var_2=None, arg_2=None):
        """ Returns an numpy array with {0,1} according to the variable array
        *var_1* and the argument array *arg_1*. It checks element wise if
        *var1* equals *arg1*, where *var1* is a array like object and *arg1*
        a scalar. It can be extended for a second variable array *var_2* and a
        second argument array *arg_2*. The function is used for computing the
        indicator function of the *indicator_model()* function.

        """
        # Create full array with the argument *arg_1*
        full_array_1 = np.full(self.n_size, arg_1, dtype='int32')
        # We test for the equality then elementwise.
        equal_1 = np.equal(full_array_1, var_1)
        if var_2 is None and arg_2 is None:
            return 1 * equal_1
        else:
            full_array_2 = np.full(self.n_size, arg_2, dtype='int32')
            equal_2 = np.equal(full_array_2, var_2)
            return 1 * np.logical_and(equal_1, equal_2)

    def indicator_model(self):
        """Returns the covariant matrix *x_matrix* (shape = [n_size, 5]) and the
        target variable *y_vector* (shape = [n_size]) of the M3 Model from
        :cite:`buhlmann2003bagging` as a numpy array for the values specified
        in the class instance.

        Note that this data generating process was *not* used in the final paper,
        but offers an interesting comparison for the reader and was thus added
        later to the appendix.

        See :cite:`buhlmann2003bagging` for the exact functional form.
        """
        # Initialize the variables covariante matrix.
        # Note we always add +1 to the desired variable due to the
        # properties of numpy arrays (last value excluded).
        x_1 = self.random_state.randint(0, 1 + 1, size=self.n_size)
        x_2 = self.random_state.randint(0, 1 + 1, size=self.n_size)
        x_3 = self.random_state.randint(0, 3 + 1, size=self.n_size)
        x_4 = self.random_state.randint(0, 3 + 1, size=self.n_size)
        x_5 = self.random_state.randint(0, 7 + 1, size=self.n_size)

        # Initialize the regression function of the indicator model.
        f_x = (
            1 *
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
            self._indicator_function(x_2, 0, x_3, 1)
        )

        # Create the covariante matrix by stacking the corresponding variables
        x_matrix = np.stack((x_1, x_2, x_3, x_4, x_5), axis=1)

        # Add an error term to *f_x* if this is desired.
        y_vector = self._set_y(f_x)

        return x_matrix, y_vector
