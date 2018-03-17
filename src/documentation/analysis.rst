.. _analysis:

************************************
Main Calculations and Simulations
************************************

Documentation of the code in *src.analysis*. This is the core of the project and contains all fundamental
calculations and simulations of the final paper.


Theoretical Calculations
=========================
Documentation of the code in *src.analysis.theory_simulation*.
All modules listed below are used for the theoretical simulations and calculations related to bagging and subagging.

Simulating the convergence for the finite sample case
*****************************************************

The finite sample sample simulation can be found under *src.analysis.theory_simulation.calc_finite_sample*.

.. automodule:: src.analysis.theory_simulation.calc_finite_sample
    :members:

Calculations for the introductory example
*****************************************

The calculations for the toy example can be found under *src.analysis.theory_simulation.calc_toy_example*.

.. automodule:: src.analysis.theory_simulation.calc_toy_example
    :members:

Calculations for stump predictors using subagging
*************************************************

The calculations for the subagging of stump predictors can be found under *src.analysis.theory_simulation.calc_normal_splits*.

.. automodule:: src.analysis.theory_simulation.calc_normal_splits
    :members:


Main Simulations
================
Documentation of the code in *src.analysis.main_simulation*.
All modules listed below use the ``MonteCarloSimulation`` Class in *src.analysis.montecarlosimulation*.
I define the simulation setup and the data generating process as a class instance.
Using the functions of the class, I then analysis changes in the bagging parameters for an else constant
simulation set up.
For more details regarding the general simulation set-up see :ref:`model_code`.

The Case of Subagging
*********************

The module with the simulations for subagging can be found under *src.analysis.main_simulation.calc_simulation_subagging*.

.. automodule:: src.analysis.main_simulation.calc_simulation_subagging
    :members:


Varying the Number of Bootstrap Iterations
******************************************

The module regarding the simulations of the convergence of bagging towards a stable value can be found under *src.analysis.main_simulation.calc_simulation_convergence*.

.. automodule:: src.analysis.main_simulation.calc_simulation_convergence
    :members:


Varying the Complexity of the Regression Trees
**********************************************

The module regarding the simulations of the model complexity can be found under *src.analysis.main_simulation.calc_simulation_tree_depth*.


.. automodule:: src.analysis.main_simulation.calc_simulation_tree_depth
    :members:


Real Data Simulations using the Boston Housing Data
===================================================
Documentation of the code in *src.analysis.real_data_simulation*.


Following the simulation set-up by :cite:`Breiman1996`, we show that the method also works, when applied to
real data.
As Bagging applied to Regression Trees is mostly used for prediction purposes, we pick a
classical prediction problem data set, namely the Boston Housing Data Set.
It was obtain from the scikit-learn library ``sklearn.datasets``.

The module with the real data simulations can be found under *src.analysis.real_data_simulation.calc_boston*.

.. automodule:: src.analysis.real_data_simulation.calc_boston
    :members:
