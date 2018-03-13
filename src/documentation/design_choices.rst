.. _design_choices:

******************************************************
Overview and explanations for different design choices
******************************************************
In the following we will give an overview over different general and overreaching design choices that have been made within this project.
We will focus on those that might seem unintuitive at first glance and give a short explanation to those.
:cite:`Gaudecker2014`.
Within each subsection of the documentation further design choices are explained in more detail.


Choices of classes instead of simple modules with functions
===========================================================

The heart of the simulation study and the implementation of the bagging algorithm
is located in the :ref:`model_code` folder. There you can find three different
modules which all host one class each.
Those three classes are the foundation for all simulation studies run in
Section 5 of the paper.

The ``DataSimulation`` class consists of three different
data generating processes (DGP). You define your specification (sample size, noise etc.) as parameters of a class
instance and then call the DGP of interest by the functions of the class instance.
The idea is that we can define a base object (the class instance) such that we can create
easily different functions, which share the specifications like the sample size and only differ
in the functional form of f(X).
Parts of the simulation have the objective to show that Bagging, when applies to
Regression Trees, works with a wide variety of functional forms. Hence we only
want to vary the functional forms and keep all other specifications constant, when comparing Bagging between them.
The class implementation is therefore a easy and straightforward choice, as we can
create one instance with the parameters of interest and then easily apply Bagging
to the different functions that are hosted in the class.

The ``BaggingTree`` class is the implementation of the Bagging Algorithm.
A class is here straightforward as we want to train the algorithm on data and
then pass the trained algorithm as a object.
Later, we can then use this trained object to make a new prediction.
While other implementations, that do not use object oriented programming, would
have been possible as well, this solution is the most straightforward.
Also it follows the practice of common packages like ``statsmodels`` or
``scikit-learn`` with this.

Also for the ``MonteCarloSimulation`` we have chosen a class implementation.
It is used to perform the three simulations which can be found in Section 5 of
the paper. It simulates the MSPE and its decomposition.
As the general simulation setup remains the same for all three simulations, it
made sense to define a separate module.
Eventually we opted for a class implementation. The implementation of this
follows the following logical concept:

As the parameters on the class instance level we pass the specifications for
the DGP and the general simulation setup (like the number of Monte Carlo iterations).
Within the two functions **calc_mse()** and **calc_mse_all_ratios()** of the class, we then pass
parameters that are specific to the Bagging Algorithm.
In general, we are interested in the effect of the variation
of different bagging algorithm parameters, while keeping the DGP and the simulation
setup constant.
Hence, the idea of the concept of the simulations described in :ref:`analysis` is to define
a class instance and then loop over different bagging algorithm parameters to
observe the changes in MSPE and its decomposition.
Doing this in the class implementation is particularly easy as we define only one
class instance for each simulation at the beginning and then use the functions **calc_mse()** and **calc_mse_all_ratios()**
with different bagging parameters within the loop.
This way we can concentrate on the changes in the Bagging parameters and have a
clear structure in the simulations.



Different RandomSeeds within the Simulations
============================================

Within the ``MonteCarloSimulation`` class, we have to pass a list of four
integer numbers, that will be used to define different RandomStates.
Each of those will be used for a different random process in the simulation.
Those different random processes should not share a common seed to achieve
meaningful results.
Namely, the four different RandomStates are:

- The RandomState for the Bagging Algorithm.
- The RandomState to create test samples.
- The RandomState to create training samples.
- The RandomState to create the new noise terms.

It is worth noting that defining ``np.random.seed()`` will not change any of those
random seeds, as they are encapsulated objects. We do not want to define a random seed at the beginning of the simulation for
a few very particular reasons:

1. We define the algorithms and data generating processes as classes that should
also be usable outside of this specific simulation setup. Also when using one of the
classes outside this simulation results should be reproducible.
Hence, we want to give the user the possibility to define a random seed for each
class separately.
Reseeding the whole numpy process by ``np.random.seed()`` within an estimator or utility class
should be considered bad practice. The reason for this is that it would change the seed
for all numpy.random functions in the module of usage.
Hence, the choice to define a new `container <https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html>`_ for the random number generator within
each class yields a high degree of encapsulation and avoids unsafe reseeding.
It is worth noting that this procedure is also considered best practice for popular
packages like ``scikit-learn``. See for instance the implementation of the bagging
algorithm by `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/bagging.py#L399>`_.


2. The second reason why define new RandomState containers instead of defining
one random seed at the beginning of the simulation is more of practical reason.
The simulation for bagging Regression Trees is computationally expensive. For
each simulation iteration we have to draw bootstrap samples and fit Regression
Trees to those bootstrap sample.
Because of this in combination with limited computational resources and as we wanted to follow the simulation
specifications of :cite:`Buhlmann2002`, we limited our number of Monte Carlo
Iterations to 100.
Now, as we want to observe the changes in MSPE decomposition, while keeping the
data simulated for each iteration constant.
Drawing truly new data sets for each iteration is no problem in the case of a high
number of monte carlo iterations by the (weak) Law of Large Numbers.
In our case however it is not feasible, as our number of iterations is so small.
The graphs would be very uneven and we could not show the clear intuition that we
give in the paper.
Hence, we redefine the RandomState for each iteration newly to draw the same samples
for each iteration.
Doing this by ``np.random.seed()`` should be considered unsafe in our simulation
due to the interplay of various random processes that should remain independent from another.
So we avoid this kind of reseeding by simply redefining the RandomState at each iteration to achieve smooth plots.
While this might not be the most elegant solution, we can obtain the results desired and replicate Bühlmann and Yu (2002)
with their simulation setup.
Another option would have been to draw all simulated data sets in advance and reuse them for each iteration.
It would yield identical results. However, we would not save computing power with this (at least not a lot), as the most time consuming
part of the simulation is the fitting process for the trees which can not be bypassed by drawing samples in advance.

For those two main reasons, we decided to define RandomState instances for each random process.
While it makes the setup a little more cumbersome, it yields a high degree of encapsulation and
thereby makes the actual simulation setup easier from programming perspective.


Speed up and why not draw samples in advance
============================================

In general simulating bagging is computationally expensive due to the very nature
of bagging. For each simulation iteration we draw a new sample and then bootstrap
samples (usually around 50) from this sample. For each bootstrap sample we then
have to fit Regression Trees on it, which we will later average.
Thus, with the parameter choices used in the paper (100 Monte Carlo Iterations
and 50 Bootstrap Sample for the Bagging Algorithm), we have to fit 5000 Regression
Trees for just one parameter specification.
In the paper we are however interested in the effect different parameter variations
have on the MSPE decomposition. Thus, for instance just for the simulation on the Convergence
of Bagging, we have to fit 250000 Regression Trees.
In an analysis with cProfile we can see that fitting Regression Trees is by far
the most computationally intensive part and consumes the great majority of
the overall run time.
Drawing the different samples however is computationally cheap in comparison.
This is why we decided to keep the design such that the new training samples are
drawn during the simulation and not in advance by a different module. Also, this way
we do not have to load all samples at the same time to the RAM, which might be more efficent.
However, we have tried different techniques to speed up the simulation process in order to
decrease the run time and/or increase the number of simulation iterations.
Small speed ups were possible by restructuring parts of the code in comparison
to the original form.
A use of C-compiler packages like Cython however did not yield any significant
improvment. The reason for this is, that it does not offer any improvement to the
most time consuming part of fitting the Regression Trees.
We use the Regression Tree implemented by ``scikit-learn``, which have naturally already
been highly optimized in Cython.
Thus, we cannot same run time with this most time consuming part anymore.

Also a parallel implementation of the Bagging Algorithm did not yield any run time
improvement. Further information on this can be found in the :ref:`model_code`
part of the documentation.


Violation of Python Conventions
=============================

We define within the ``BaggingTrees`` class, a class attribute
outside the ``__init__`` function. This violates ``python`` conventions
according to an analysis run with ``pylint``, but is of great use in the case
of bagging, as we want to pass a newly trained class instance each time after
we fit the data. We decided to keep it like this, as it is also used by other
`packages <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/bagging.py#L335>`_.
