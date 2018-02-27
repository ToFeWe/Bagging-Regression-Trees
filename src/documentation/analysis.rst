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

.. automodule:: src.analysis.theory_simulation.calc_finite_sample
    :members:

Calculations for the introductory example
*****************************************

.. automodule:: src.analysis.theory_simulation.calc_toy_example
    :members:

Calculations for stump predictors using subagging
*************************************************

.. automodule:: src.analysis.theory_simulation.calc_normal_splits
    :members:


Main Simulations
================
Documentation of the code in *src.analysis.main_simulation*.
All modules listed below use the ``MonteCarloSimulation`` Class in *src.analysis.montecarlosimulation*.
We define the simulation setup and the data generating process as a class instance.
Using the functions of the class, we then analysis changes in the bagging parameters for an else constant
simulation set up.
For more details regarding the general simulation set-up see :ref:`model_code`

The Case of Subagging
*********************

.. automodule:: src.analysis.main_simulation.calc_simulation_subagging
    :members:


Varying the Number of Bootstrap Iterations
******************************************

.. automodule:: src.analysis.main_simulation.calc_simulation_convergences
    :members:


Varying the Complexity of the Regression Trees
**********************************************

.. automodule:: src.analysis.main_simulation.calc_simulation_tree_depth
    :members:


Real Data Simulations using the Boston Housing Data
===================================================
Documentation of the code in *src.analysis.real_data_simulation*.


Following the simulation set-up by Breiman (1996), we show that the method also works, when applied to
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
