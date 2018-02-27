"""

This module performs the simulations of the MSPE and its decomposition into squared-bias, variance and noise, for the
Bagging Algorithm as described in the paper:

In all simulations we use the following procedure:
i. Generate a test sample, without error term, according to the data generating processes of
interest. This will be constant for the whole simulation study. All predictions will be made
on this sample.10
ii. For each simulation iteration we follow this procedure:
    (a) Draw new error terms for the test sample.
    (b) Draw a new training sample with regressors and error terms.
    (c) Fit the predictor (Tree, Bagging, Subagging) to the generated training data.
    (d) Using this new predictor make a prediction into the fixed test sample and save the
    predicted values.
iii. We compute the MSPE, squared bias and variance for the given predictor at the input point X = x0 with x0 being the test sample generated in (i).



Within one class instance we define all parameters that are relevant to the data generating process (DGP) and the simulation set-up.
Parameters that are specific to the Bagging Algorithm are defined in the functions. The idea is to define one class
instance and then loop over different bagging parameters for the same instance, keeping the DGP and the simulation set-up
constant.
The function calc_mse() computes the MSPE and its decomposition for one specification and the calc_mse_all_ratios() for a
range of subsampling ratios.

"""

# Import the needed packages
import numpy as np
from src.model_code.baggingtree import BaggingTree
from src.model_code.datasimulation import DataSimulation


class MonteCarloSimulation:
    """
    A  class that implements a Monte Carlo Simulation for the Bagging Algortihm.

    Parameters
    ----------
    random_seeds: tuple of size 4 consisting of int or None, optional (Default: (None, None, None, None))
        Specify the random seeds that will be used for the simulation study. We have to use different random seeds, as
        we define different RandomState instances for each part of the simulation.

        random_seeds[0]: Defines the RandomState for the noise term draw
        random_seeds[1]: Defines the RandomState for the BaggingTree class
        random_seeds[2]: Defines the RandomState for the training sample draws
        random_seeds[3]: Defines the RandomState for the test sample draw

        One random_seed is used to specify the RandomState for numpy.random.
        It is shared accros all functions of the class.

        IMPORTANT: This random seed is fixed for a specific instance, as it specifies a new RandomState for all numpy functions
        used in the class. As a result this random_seed is *not* overwritten by numpy
        random seeds that are defined outside of specific class instance. The reason for
        this is that it makes reproducibility easier accross different simulations and
        modules.
        Note however that the downside is, that we have to specify for each class (each instance)
        a different random seed and it is not possible to specify one random seed at the beginning
        of the whole simulation, as this will define the RandomState within each class.

    noise: int, float, optional (Default=1.0)
        The variance of the error term that is used for the data generating
        processes.
        The default of *noise* = 1.0 indicates that we draw without an error term
        that is standard normally distributed.

    n_test_train: tuple of size 2 with int, optional(Default= (500, 500))
        Specify the sample size of the test sample and the training samples.

        n_test_train[0]: Defines the size for the test sample
        n_test_train[1]: Defines the size for the training samples

    data_process: string, optional (Default="fiedman")
        Defines which data generating process we use.
        Other options are "linear" and "indicator".

    """

    def __init__(self,
                 n_repeat=100,
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
        self.test_simulation = DataSimulation(
            n_size=self.n_test,
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

    def calc_mse(
            self,
            ratio=1,
            bootstrap=True,
            min_split_tree=2,
            b_iterations=50):
        """
        A  function to simulate he MSPE decomposition for one specific specification of the Bagging Algorithm applied to Regression Trees.
        The simulation set up and the data generating process is given by the respective class instance. We want to compare
        the output of this function with respect to variations in the Bagging parameters.

        Returns a numpy array of size 4 with the MSPE decomposition:
            array[0]: Simulated MSPE
            array[1]: Simulated squared bias
            array[2]: Simulated variance
            array[3]: Simulated noise

        Parameters
        ----------
        ratio: float, optional (Default=1.0)
            The sample size used for the simulation procedure. Each sample we draw for the Bagging Algorithm will be of size
            math.ceil(n_observations * self.ratio).

        min_split_tree: int, optional (Default=2)
            The minimal number of observations within a terminal node of the Regression Trees to be
            considered for a split that are used in the simulation.
            Use this to control for the complexity of the Regression Tree.

            Must be greater than 2.

        b_iterations: int, optional (Default=50)
            The number of bootstrap iterations used to construct the bagging/subagging predictor in the simulation.


        bootstrap: bool, optional(Default=True)
            Specify if the you use the standard bootstrap (Bagging) or m out of n bootstrap (Subagging).

            Default=True implies that we use Bagging.
        """

        # Create the instance of the bagging algorithm class, with the given
        # parameters, that will be used for the rest of the simulation run.
        bagging_instance = BaggingTree(random_seed=self.random_seed_fit,
                                       ratio=ratio, bootstrap=bootstrap,
                                       b_iterations=b_iterations,
                                       min_split_tree=min_split_tree)

        # To make results comparable and to get a smooth plot (we have
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
        y_se_all = np.ones((self.n_repeat, self.n_test)) * np.nan

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
            y_se_all[i, :] = (y_test - predictions[:, i]) ** 2

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

    def calc_mse_all_ratios(
            self,
            n_ratios=10,
            min_ratio=0.1,
            max_ratio=1,
            min_split_tree=2,
            b_iterations=50):
        """
        A  function to simulate he MSPE decomposition for a range of subsampling fractions for
        one specification of the Subagging Algorithm applied to Regression Trees.
        The simulation set up and the data generating process is given by the respective class instance. We want to compare
        the output of this function with respect to variations in the Bagging parameters and the variation between the
        subsampling fractions.
        Note that by defaul we use subsampling instead of the standard bootstrap.

        The range of subsampling ratios is created by np.linspace(*min_ratio*, *max_ratio*, *n_ratios*).

        Returns a numpy array of shape = [n_ratios, 4] with the MSPE decomposition for the *n_ratios* different subsamling
        ratios:
            array[:,0]: Simulated MSPE for all subsampling ratios
            array[:,1]: Simulated squared bias for all subsampling ratios
            array[:,2]: Simulated variance for all subsampling ratios
            array[:,3]: Simulated noise for all subsampling ratios

        Parameters
        ----------
        n_ratios: int, optional (Default=10)
            The number of subsampling fractions we want to consider for the simulation.

        min_ratio: float, optional (Default=0.1)
            The minimal subsampling fraction to be considered.

        max_ratio: float, optional (Default=1.0)
            The maximal subsampling fraction to be considered.

        min_split_tree: int, optional (Default=2)
            The minimal number of observations within a terminal node of the Regression Trees to be
            considered for a split that are used in the simulation.
            Use this to control for the complexity of the Regression Tree.

            Must be greater than 2.

        b_iterations: int, optional (Default=50)
            The number of bootstrap iterations used to construct the subagging predictor in the simulation.

        """
        # Array must be of length four: MSPE, Bias, Variance, Noise
        array_length = 4
        # Create a range of subsampling ratios.
        ratiorange = np.linspace(min_ratio, max_ratio, n_ratios)

        # Create an array to save simulation for each ratio.
        output_array_subagging = np.ones((n_ratios, array_length)) * np.nan

        # We loop over all ratios and save the results to an array.
        for index, ratio in enumerate(ratiorange):
            output_array_subagging[index,
                                   :] = self.calc_mse(ratio=ratio,
                                                      bootstrap=False,
                                                      min_split_tree=min_split_tree,
                                                      b_iterations=b_iterations)
        return output_array_subagging
