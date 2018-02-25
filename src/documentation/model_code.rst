.. _model_code:

**********
Model code
**********


The directory *src.model_code* contains source files for the Bagging Algorithm,
the Data Generating Process and a module that is used to perform the Monte Carlo Simulations.


The ``BaggingTree`` class
=========================
The ``BaggingTree`` class is the implementation of the Bagging
Algorithm applied to Regression Trees.
For Regression Trees the implementation of ``sklearn`` is used and was not implemented
within this algorithm.

.. automodule:: src.model_code.baggingtree
    :members:


The ``DataSimulation`` class
============================
The ``DataSimulation`` class contains a collection of data generating processes.
We define the characteristic that are shared among the different functions in the
class instance.
Then different functions can be called to create different simulated data sets that share common attributes.
This is helpful as we want to compare the effectiveness of the Bagging Algorithm among
different functional forms while keeping attributes like the sample size or the noise
constant across different regression functions.

.. automodule:: src.model_code.datasimulation
    :members:

The ``MonteCarloSimulation`` class
==================================
The ``MonteCarloSimulation`` class implements the Monte Carlo simulation as it was
used in the **Simulation** part of the paper.
We picked this simulation procedure as we wanted to emphasis the decomposition of the
mean squared prediction error at a new input point into Bias and the Variance
but also the irreducible Noise term.


.. automodule:: src.model_code.montecarlosimulation
    :members:
