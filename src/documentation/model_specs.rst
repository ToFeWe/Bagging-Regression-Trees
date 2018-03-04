.. _model_specs:

********************
Model specifications
********************

The directory *src.model_specs* contains `JSON <http://www.json.org/>`_ files with model specifications.
They are used across different parts of the model to specify the simulations/calculations or make the
plotting uniform across different modules. We decided to split those specification in a lot of
different files to make it easier to change only certain parts of the specifications
without having to rerun the whole code in waf.

Overview for JSON files
=======================
All JSON files (except *dgp_models.json*) are used to define a dictionary in python. Below we will give
a short descriptions to all json files (also referred to as dictionary) and its
keys.
The default values are all inline with the descriptions in the final term paper
and hence omitted here.

boston_settings.json
********************
The dictionary defines the simulation set-up that is specific to the boston simulation.

Keys
----
ratio_test: float
  Ratio for the test sample
ratio_train: float
  Counterpart to *ratio_test*
random_seed_split: int
  Starting point for the random seeds for the test_train_split
random_seed_fit: int
  Random seed for the fitting procedure
  
convergence_settings.json
*************************
The dictionary defines the simulation set-up that is specific to the convergence of the Bagging Algorithm.

Keys
----
max_bootstrap: int
  Maximum number of bootstraps in the range to be considered
min_bootstrap: int
  Minimum number of bootstraps in the range to be considered
steps_bootstrap: int
  Steps in the range between *min_bootstrap* and *max_bootstrap*
converged_bootstrap: int
  A large value of bootstrap iterations to visualize the convergence

dgp_models.json
***************
List of available data generating processes. Note that only 'linear' and
'friedman' were eventually used in the term paper.


finite_sample_settings.json
***************************
The dictionary that defines the simulation set-up for the finite sample case.

Keys
----
n_repeat: int
  Number of Monte Carlo repetitions
n_list: list
  List with the sample sizes to be considered
mu: int, float
  True mean of the population
sigma: int, float
  Standard deviation
b_iterations: int
  Number of bootstrap iterations
x_gridpoints: int
  Number of gridpoints
x_min: int
  Minimum gridpoint
x_max: int
  Maximal gridpoint
random_seed: int
 Random seed for the simulation

general_settings.json
*********************
The dictionary is shared across various simulations and defines the overall simulation set-up.

Keys
----
n_repeat: int
  Number of Monte Carlo repetitions
n_test_train: list
  List with the test and train size
noise: int, float
  Standard deviation of the error term for the data generating process
b_iterations: int
  Number of bootstrap iterations
min_split_tree: int
  Governs the tree depth. Lower values imply more complex Regression Trees
random_seeds: list
  List of random seeds used. Note: We don't reseed but define different RandomState instances with those.
BAGGING_RATIO: constant at 1
  Subsampling ratio for bagging. Do not change!

general_settings_small.json
***************************
Same as *general_settings.json* but smaller specification. Can be used for testing.


normal_splits_settings.json
***************************
The dictionary defines the calculation set-up that is specific to the stump predictor simulation.

Keys
----
c_gridpoints: int
  Number of gridpoints for c
c_min: int
  Minimum gridpoint
c_max: int
  Maximal gridpoint
a_array: dictionary
  Consists of keys that define the subsampling ratios we want to consider.
  The value of the first key has to be equal to 1.
  The other key values are defined as lists, where list[0] = numerators and
  list[1] = denominator of the subsampling fraction.
gamma: float
  Rate of convergence



settings_plotting.json
**********************
The dictionary contains all plotting specifications that are shared across various modules.

Keys
----
style: string
  Matplotlib stlye that is used for all plots
figsize: list
  List that defines the figure sizes
figsize_theory: list
  List that defines the figure sizes in the theory part
colors: dictionary
  Dictionary for uniform colors across figures
ls: dictionary
  Dictionary for uniform line style across figures

subagging_settings.json
***********************
The dictionary defines the simulation set-up that is specific to the subagging simulation.

Keys
----
n_ratios: int
  Number of subsampling ratios to be considered
max_ratio: int, float
  Maximal subsampling ratio
min_ratio: int, float
  Minimal subsampling ratio

toy_example_settings.json
*************************
The dictionary defines the calculation set-up that is specific to the introductory simulation.

Keys
----
c_gridpoints: int
  Number of gridpoints
c_min: int, float
    Minimal gridpoint
c_max: int, float
  Maximal gridpoint

tree_depth_settings.json
************************
The dictionary defines the simulation set-up that is specific to the tree depth simulation.

Keys
----
min_split: int
  Minimal split minimum for terminal nodes
max_split: int
  Maximal split minimum for terminal nodes
steps_split: int
  Steps within the range
