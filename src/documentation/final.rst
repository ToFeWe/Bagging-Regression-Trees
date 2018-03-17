.. _final:

************************************
Visualization and results formatting
************************************

Documentation of the code in *src.final*.
Below you can find the documentation to all modules, which create the figures and table for the final paper.
The output folder for all figures is *bld.out.figures* and *bld.out.tables* for the table.


Figures for the Theory Simulations and Calculations
===================================================
Documentation of the code in *src.final.theory_simulation*.


Figure 2 - Simulating the convergence of the predictor
******************************************************

.. automodule:: src.final.theory_simulation.plot_finite_sample
    :members:

Figure 3 - Bias, Variance and MSE of the predictor in the introductory example
*******************************************************************************

.. automodule:: src.final.theory_simulation.plot_toy_example
    :members:

Figure 4 - Bias, Variance and MSE for the stump predictor
*********************************************************

.. automodule:: src.final.theory_simulation.plot_normal_splits
    :members:



Figures and Table for the Main Simulations
==========================================
Documentation of the code in *src.final.main_simulation*.

Table 1 - The bagged Tree compared to the unbagged Tree
*******************************************************

.. automodule:: src.final.main_simulation.table_bagging.py
    :members:

Figure 5 - The bagging Estimator convergences towards a stable value
*********************************************************************

.. automodule:: src.final.main_simulation.plot_simulation_convergence
    :members:

Figure 6 - The effectiveness of Subagging compared to Bagging
**************************************************************

.. automodule:: src.final.main_simulation.plot_simulation_subagging
    :members:

Figure 7 - The effect of varying the Tree depth
***********************************************

.. automodule:: src.final.main_simulation.plot_simulation_tree_depth
    :members:

Figure for the Real Data Simulation
===================================
Documentation of the code in *src.final.real_data_simulation*.

Figure 8 - Bagging and Subagging applied to real data
******************************************************

.. automodule:: src.final.real_data_simulation.plot_boston
    :members:
