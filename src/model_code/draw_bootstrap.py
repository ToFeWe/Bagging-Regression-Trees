"""
Module to create a bootstrap sample, which will later be used
to create the bagging algorithm

"""
import numpy as np
import math
from sklearn.tree import DecisionTreeRegressor


class BaggingTree:
    """Implements a Bagging Predictor for Regressions Trees"""

    def __init__(
            self,
            random_seed=None,
            ratio=1.0,
            bootstrap=True,
            B_iterations=1,
            min_split_tree=2):
        self.random_seed = random_seed
        self.ratio = ratio
        self.bootstrap = bootstrap
        self.B_iterations = B_iterations
        self.min_split_tree = min_split_tree
        self.random_state = np.random.RandomState(self.random_seed)

    def _draw_sample(self, X, y):
        """Draws sample of the given data. Use *self.ratio* and *self.bootstrap*
        to specify if you want to draw with replacement (Bootstrap) and how large
        your sample should be relative to the orginal data.
        Note: Default values *self.ratio*=1 and *self.bootstrap*True indicate
        that we draw a sample for bagging.

        """

        # Number of observations in data set
        n_observations = X.shape[0]
        # Number of observations for each draw
        draw_size = math.ceil(n_observations * self.ratio)

        # Draw array of integers with/without replacement - those will be the rows
        # for the bootstrap/subsample sample
        if self.bootstrap:
            obs_range = self.random_state.choice(
                n_observations, size=draw_size, replace=True)
        else:
            obs_range = self.random_state.choice(
                n_observations, size=draw_size, replace=False)

        # Create the draw for both arrays *X* and *y* accordig to the
        # observation range *obs_range* that was created beforehand.
        X_draw = X[obs_range, :].copy()
        y_draw = y[obs_range].copy()

        return X_draw, y_draw

    def fit(self, X, y):
        # Define list of estimators that will be fit to the different Bootstrap
        # sample.
        self.tree_estimators = []

        # The actual Bagging algorithm follows. This step is repeated
        # *B_iterations* times, which is the number of Bootstrap iterations.
        for sample in range(self.B_iterations):
            # Draw a new bootstrap sample
            X_draw, y_draw = self._draw_sample(X, y)

            # We create a new tree instance for each iteration.
            tree = DecisionTreeRegressor(
                max_depth=None,
                min_samples_split=self.min_split_tree,
                random_state=self.random_seed)

            # Fit Regression Tree to the Bootstrap Sample.
            fitted_tree = tree.fit(X_draw, y_draw)

            # Append the fitted tree to the list, that contains all Regression
            # Trees
            self.tree_estimators.append(fitted_tree)
        # We return *self*, as we want be able to pass a trained instance
        return self

    def predict(self, X):

        # Number of observations for which we make a prediction
        n_observations = X.shape[0]

        # Initialize the array of predictors of the *B_iterations* Regression
        # Trees given the sample size *n_observations*.
        predictions = np.ones((self.B_iterations, n_observations)) * np.nan

        # Get prediction values for each tree.
        for i in range(self.B_iterations):
            predictions[i, :] = self.tree_estimators[i].predict(X)

        # Compute the mean over all *B_iterations* predictions for each
        # observation in X.
        bagging_estimate = predictions.mean(axis=0)

        return bagging_estimate
