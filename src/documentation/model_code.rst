.. _model_code:

**********
Main Algorithms and Model code
**********


The directory *src.model_code* contains source files for the Bagging Algorithm,
the Data Generating Process and a module that is used to perform the Monte Carlo Simulations.


The ``BaggingTree`` class
=========================
The ``BaggingTree`` class is the implementation of the Bagging
Algorithm applied to Regression Trees.
For Regression Trees the implementation of ``sklearn`` is used and was not implemented
within this algorithm.

Note that this Implementation of the Bagging Algorithm does *not* run in parallel  even though it can be considered as
*embarrassingly parallel*, as also noted by :cite:`Breiman1996` in his pioneer paper.

A runtime analysis using the Python package cProfile showed that if I parallel the algorithm with the
Python package ``joblib``, the run time is *higher* then in this current version.
One reason for this is that I restrict the number of bootstrap iterations to 50 and the sample size to 500 in the majority of the paper.
The overhead created by launching and managing multiple threads is higher than the actual
runtime gain processing the bagging operation separately, when there are only comparabily few bagging iterations.
Only when I increase the number of bootstrap iterations considerably (e.g. 500 iterations) a parallel execution becomes
profitable in terms of runtime.
Hence, for the parameter choices I consider in this paper, a parallel execution turned out not to be profitable.
Note however that given a different parameter set (more bootstrap iterations/larger sample size), paralleling the bagging
algorithm is liked to be desired.




.. automodule:: src.model_code.baggingtree
    :members:


The ``DataSimulation`` class
============================

.. automodule:: src.model_code.datasimulation
    :members:

The ``MonteCarloSimulation`` class
==================================
The ``MonteCarloSimulation`` class implements the Monte Carlo simulation for a given set of parameters as it was
used in the **Simulation** part of the paper.
We picked this simulation procedure as we wanted to emphasis the decomposition of the
mean squared prediction error at a new input point into Bias and the Variance
but also the irreducible Noise term.
It is used in the calculation modules in *src.analysis.main_simulation*, where I consider different parameter variations for the
Bagging Algorithm to observe changes in the MSPE, Bias and the Variance.
The parameters that are specific to the data generating process are defined in the class instance.
Parameters for the Bagging Algorithm are defined in the functions.



.. automodule:: src.model_code.montecarlosimulation
    :members:
