.. _analysis:

************************************
Main Calculations and Simulations
************************************

Documentation of the code in *src.analysis*. This is the core of the project.



Theoretical Calculations
=========================
Documentation of the code in *src.analysis.theory_simulation*.

Simulating the convergence for the finite sample case
*****************************************************

Without choosing a dynamic environment for x, the estimator developed by X-X and
illustrated in our paper stabilizes by the (weak) Law of Large Numbers.
We simulate this here for a range of sample sizes for a given mean and variance,
assuming that Y follows a Gaussian distribution.

.. automodule:: src.analysis.theory_simulation.calc_finite_sample
    :members:

Calculations for the introductory example
*****************************************
Given the choice of the appropriate environment of x, the estimator does not
stabilizes even asymptotically and we can illustrate the effects of bagging on
it.

.. automodule:: src.analysis.theory_simulation.calc_toy_example
    :members:

Calculations for stump predictors using subagging
*************************************************
Replacing the bootstrap procedure by a subsampling scheme, we can here calculate
upper bounds for the Variance and the Bias of stump predictors following X-X-

.. automodule:: src.analysis.theory_simulation.calc_normal_splits
    :members:


Main Simulations
================
Documentation of the code in *src.analysis.main_simulation*.

The Case of Subagging
*********************
Varying the subsampling ratio, we perform for each of those variations a Monte Carlo Simulation
to obtain the MSPE and its decomposition. Therefore we use the ``MonteCarloSimulation`` class.
We create a class instance which specifies the data generating process and then loop
over different subsampling fractions using the pre specified class instance.
We compare those results to Bagging and unbagged Regression Trees.
Note that by definition subagging with the subsampling ratio equal to 1, is equivalent to an unbagged predictor.

.. automodule:: src.analysis.main_simulation.calc_simulation_subagging
    :members:


Varying the Number of Bootstrap Iterations
******************************************
We consider only Bagging to show that when we increase the number of bootstrap iterations the predictor converges.
The same logic in implementation as before applies here.

.. automodule:: src.analysis.main_simulation.calc_simulation_convergences
    :members:


Varying the Complexity of the Regression Trees
**********************************************
We simulate effect of varying the Tree depth for bagging and unbagged Trees.
The same logic in implementation as before applies here.

.. automodule:: src.analysis.main_simulation.calc_simulation_tree_depth
    :members:


Real Data Simulations using the Boston Housing Data
=====================
Documentation of the code in *src.analysis.real_data_simulation*.


Following the simulation set-up by X-X, we show that the method also works, when applied to
real data.
As Bagging applied to Regression Trees is mostly used for prediction purposes, we pick a
classical prediction problem data set, namely the Boston Housing Data Set.
It was obtain from the scikit-learn library ``sklearn.datasets``.
The simulation is parallelized, which increases the speed drastically. The functions are tested and
work well behaved, even though it might not be the most elegant and prettiest solution.

Note that we use the ``joblib`` library to run the simulation as a pipeline job. This package
should have been installed, when setting up the conde eniroemnt. If you run into any error
install it from https://pythonhosted.org/joblib/
or contact s6towern@uni-bonn.de.


.. automodule:: src.analysis.real_data_simulation.calc_boston
    :members:
