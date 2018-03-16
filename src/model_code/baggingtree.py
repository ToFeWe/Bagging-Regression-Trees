"""

This module implements the Bagging Algorithm used for the main simulations of
this paper. To use it, you first define a class instance that specifies the
parameters for the algorithm. Then use the fit() function to fit the algorithm
to a training sample. Predictions on a new sample can be made using the predict()
function.

"""
import math
import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class BaggingTree:
    """
    A  class that implements the Bagging Algorithm applied to Regression Trees.
    For the Regression Trees we use the implementation of scikit-learn.

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

    ratio: float, optional (Default=1.0)
        The sample size for the subsampling procedure. Each sample we draw for
        the algorithm will be of size math.ceil(n_observations * self.ratio).

        Needs to be greater than 0 and smaller than 1.

        In accordance with the theoretical treatment in the paper, one would
        want to choose *ratio<1.0* for *bootstrap=False* (Subagging) and
        *ratio=1.0*  for *bootstrap=True* (Bagging).


    min_split_tree: int, optional (Default=2)
        The minimal number of observations that can be within a terminal node
        of the Regression Trees to be considered for a split.
        Use this to control for the complexity of the Regression Tree.

        Needs to be greater than 1.

    b_iterations: int, optional (Default=50)
        The number of bootstrap iterations used to construct the bagging/subagging
        predictor.

        Needs to be greater than 0.

    bootstrap: bool, optional(Default=True)
        Specify if the you use the standard bootstrap (Bagging) or m out of n
        bootstrap (Subagging).

        Default=True implies that we use Bagging.
    """

    def __init__(
            self,
            random_seed=None,
            ratio=1.0,
            bootstrap=True,
            b_iterations=50,
            min_split_tree=2):
        # Set and check the inputs.
        self._set_ratio(ratio)
        self._set_bootstrap(bootstrap)
        self._set_b_iterations(b_iterations)
        self._set_min_split_tree(min_split_tree)

        # No function to check the seed as it is done directly by numpy.
        self.random_state = np.random.RandomState(random_seed)
        # Fot the Tree we need the RandomState directly.
        self.random_seed = random_seed

    def _set_ratio(self, ratio):
        """ A function to check if *ratio* is specified correctly. """
        assert np.issubdtype(
            type(ratio), np.floating), ' The *ratio* needs to be of type float.'
        assert 1 >= ratio > 0, \
            'It is required that 1 >= *ratio* > 0. You provided ratio={}'.format(ratio)
        self.ratio = ratio

    def _set_bootstrap(self, bootstrap):
        """ A function to check if *bootstrap* is specified correctly. """
        assert isinstance(bootstrap, bool), \
            ('*bootstrap* needs to be of type *bool*. '
             'The provided value is of type {}'.format(type(bootstrap)))
        if self.ratio != 1.0 and bootstrap:
            warnings.warn('You are using subsampling without replacement'
                          '(default for bagging) but *ratio* != 1.'
                          ' Hence you use neither the m out of n bootstrap'
                          ' nor the standard bootstrap.')
        self.bootstrap = bootstrap

    def _set_b_iterations(self, b_iterations):
        """ A function to check if *b_iterations* is specified correctly. """
        assert np.issubdtype(type(b_iterations), np.integer) and b_iterations > 0, \
            ('*b_iterations* need to be an integer greater than zero.'
             ' You provided b_iteartions={}, which is of type {}.'
             ''.format(b_iterations, type(b_iterations)))
        self.b_iterations = b_iterations

    def _set_min_split_tree(self, min_split_tree):
        """ A function to check if *min_split_tree* is specified correctly. """
        assert np.issubdtype(type(min_split_tree), np.integer) and min_split_tree > 1, \
            ('*min_split_tree* need to be an integer greater than one. '
             'You provided min_split_tree={}, which is of type {}.'
             ''.format(min_split_tree, type(min_split_tree)))
        self.min_split_tree = min_split_tree

    def _draw_sample(self, x_matrix, y_vector):
        """Draws sample of the given data. Use on the class level *self.ratio*
        and *self.bootstrap* to specify if you want to draw with replacement
        (Bootstrap) and how large your sample should be relative to the original
        data.

        """

        # Number of observations in data set
        n_observations = x_matrix.shape[0]
        # Number of observations for each draw
        draw_size = math.ceil(n_observations * self.ratio)

        # Draw array of integers with/without replacement - those will be the rows
        # for the bootstrap/subsample sample.
        if self.bootstrap:
            obs_range = (
                self.random_state.choice(n_observations, size=draw_size, replace=True)
            )
        else:
            obs_range = (
                self.random_state.choice(n_observations, size=draw_size, replace=False)
            )

        # Create the draw for both, the matrix *x_matrix* and the vector *y_vector*, according
        # to the observation range *obs_range* that was created beforehand.
        x_matrix_draw = x_matrix[obs_range, :].copy()
        y_draw = y_vector[obs_range].copy()

        return x_matrix_draw, y_draw

    def fit(self, x_matrix, y_vector):
        """
        Fit the Bagging Algorithm *newly* to a sample (usually training sample)
        that consists of the covariant matrix *x_matrix* and the vector the dependent
        variable *y_vector*.

        Parameters
        ----------
        x_matrix: numpy-array with shape = [n_size, n_features] (Default: None)
            The covariant matrix *x_matrix* with the sample size n_size and
            n_features of covariants.

        y_vector: numpy-array with shape = [n_size,] (Default: None)
            The vector of the dependent variable *y_vector* with the sample size n_size

        """
        # We check the inputs for the function.
        self._check_fit(x_matrix, y_vector)

        # Define a new list of estimators that will be fit to the different Bootstrap
        # samples.
        self.tree_estimators = []

        # The actual Bagging algorithm follows. This step is repeated
        # *b_iterations* times, which is the number of Bootstrap iterations.
        for _ in range(self.b_iterations):
            # Draw a new bootstrap sample
            x_matrix_draw, y_draw = self._draw_sample(x_matrix, y_vector)

            # We create a new tree instance for each iteration.
            # Note that the random seed can be constant here for each
            # iteration.
            tree = DecisionTreeRegressor(
                max_depth=None,
                min_samples_split=self.min_split_tree,
                random_state=self.random_seed)

            # Fit Regression Tree to the Bootstrap Sample.
            fitted_tree = tree.fit(x_matrix_draw, y_draw)

            # Append the fitted tree to the list, that contains all Regression
            # Trees.
            self.tree_estimators.append(fitted_tree)
        # We return *self*, as we want be able to pass a trained instance.
        return self

    @staticmethod
    def _check_fit(x_matrix, y_vector):
        """
        A static function to check the inputs fot the *fit()* function.
        As it is static, we used the ``staticmethod`` decorator and
        dropped *self* from the attributes.

        """

        assert isinstance(x_matrix, np.ndarray), \
            'Your input x_matrix is not a numpy array. Currently only those are supported.'
        assert isinstance(y_vector, np.ndarray), \
            'Your input x_matrix is not a numpy array. Currently only those are supported.'
        assert x_matrix.ndim == 2, 'The convariant matrix *x_matrix* must be two dimensional.'
        assert y_vector.ndim == 1, 'The vector *y_vector* must be one dimensional.'

    def predict(self, x_matrix):
        """
        Make a new prediction for a **trained** class instance (using the fit()
        function first) on a new covariant matrix *x_matrix* (test sample).

        Parameters
        ----------
        x_matrix: numpy-array with shape = [n_size, n_features] (Default: None)
            The covariant matrix *x_matrix* of the new test sample with size n_size and
            n_features covariants.

        """
        # We check the inputs for the function.
        self._check_predict(x_matrix)

        # Number of observations for which we make a prediction.
        # Might differ from *n_observations* in the *fit()* function, that is
        # why it is not defined on the class level.
        n_observations = x_matrix.shape[0]

        # Initialize the array of predictors of the *b_iterations* Regression
        # Trees given the sample size *n_observations*.
        predictions = np.ones((self.b_iterations, n_observations)) * np.nan

        # Get prediction values for each tree.
        for i in range(self.b_iterations):
            predictions[i, :] = self.tree_estimators[i].predict(x_matrix)

        # Compute the mean over all *b_iterations* predictions for each
        # observation in x_matrix.
        bagging_estimate = predictions.mean(axis=0)

        return bagging_estimate

    def _check_predict(self, x_matrix):
        """ A function that checks the inputs of the *predict()* function"""

        assert hasattr(self, 'tree_estimators'), \
            ('The predict method needs a trained BaggingTree instance. First '
             'fit a BaggingTree to a training set using the *fit()* function.')
        assert isinstance(x_matrix, np.ndarray), \
            'Your input x_matrix is not a numpy array. Currently only those are supported.'
        assert x_matrix.ndim == 2, 'The convariant matrix *x_matrix* must be two dimensional.'
        assert self.tree_estimators[0].n_features_ == x_matrix.shape[1], \
            ('The number of features between sample used in the *fit()* and '
             'the *predict()* functions must be the same.')
