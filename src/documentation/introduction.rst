.. _introduction:


************
Introduction
************

This is the documentation of the term paper project 'Bagging Regression Trees' in the project module
'Econometrics and Statistics' in the winter semester 2017/2018 taught by Professor Alois Kneip and Professor Dominik Liebl at the University of Bonn.

The authors of the term paper are Robin Kraft, Tobias Felix Werner and David Zeimentz.

All code, including the implementation to waf, was written by Tobias Felix Werner.

The waf template was designed by cite:`Gaudecker2014`. Documentation for Waf and more background on this is at http://hmgaudecker.github.io/econ-project-templates/ and https://github.com/hmgaudecker/econ-project-templates


Research Question
=================
In statistical estimation, most prediction problems encounter a bias-variance tradeoff.
A class of predictors for which this is pronounced are so-called Regression Trees (cite:`Breiman1984`).
The  Bagging algorithm proposed by cite:`Breiman1996` bypasses this tradeoff by reducing
the variance of the unstable predictor, while leaving its bias mostly unaffected. In particular,
Bagging uses repeated bootstrap sampling to construct multiple versions of the same prediction
model like Regression Trees and averages over the resulting predictions.
In the paper we show that Bagging can reduce the variance of Regression Tree predictors
and thereby improves their prediction accuracy in terms of the mean squared prediction error
(MSPE).

Overview for the project
========================
The implementation of the simulations described in the term paper has been split up across various modules. The objective of this section is
to explain the concept and structure.

The part :ref:`model_code` contains explanations for all basic algorithms and functions that are used for the main parts of the simulation, which can be found in Section 5.
Amongst them is the implementation
of the Bagging Algorithm for Regression Trees in the ``BaggingTree`` class and the ``MonteCarloSimulation`` class, which is used to perform the simulations of Section 5 of the term paper.
Within :ref:`analysis` the documentation for the main calculations and simulations of the paper can be found. The analysis part has been split up into three subfolders to maintain a clear
structure for the project. The three subfolders coincide with the three computationally intensive parts of the paper, which are the calculations for the theoretical part in Sections 3 and 4,
the main simulations in Section 5 and the application of the Bagging Algorithm to real data in Section 6. As the simulation setup differs drastically between those parts, it made sense
to reflect this also in the structure of the project.
In :ref:`final` the modules for creating the figures and tables are documented. Note that here I also maintain the structure chosen in :ref:`analysis`.
All model specifications for the different simulations are defined in ``.json`` files. This way I can easily ensure that modules/simulations that should share common attributes
actually share them. An overview for the different ``.json`` files can be found in :ref:`model_specs`.
Finally, the overview of the final output files is listed in :ref:`paper`. In this folder you can also find further figures, that have not been created during the
build process.
As this project is mostly based on simulated data or no data at all, we do not have a separate folder for data management.
The real data we use in the :ref:`analysis` part was directly obtained from the python package ``scikit-learn``.

How to run parts of the code? How to replicate the output myself?
==========================================================

The term paper project has been implemented using Python and a replication template by cite:`Gaudecker2014`, which builds on the build automation tool waf.

To run the code yourself you have to follow the following steps:
  * Make sure to have Miniconda or Anaconda installed. A modern LaTeX distribution (e.g. TeXLive, MacTex, or MikTex) needs to be found on your path.
  * Download the project or clone it to your machine.
  * Move to the root directory of the project in the shell.
  * Run ``source set-env.sh`` (Linux, MacOS) or ``set-env.bat`` to set up the conda environment specific to this project.
  * Run ``python waf.py configure`` and then ``python waf.py build`` to run the whole project. Note that if you use the
    parameter specification used in the paper, this may take some time (~ 40 min with i5-6200 CPU @ 2.3 GHz, 8 GB RAM).
    If you are interested in the implementation you can also scale down the parameters of the simulations in the
    respective JSON files described in :ref:`model_specs`.

Note that its crucial to setup the conda environment as the packages chosen are tailored specifically to this project and
might differ to your conda root environment.

The replication template was obtained from https://github.com/hmgaudecker/econ-project-templates.
For further information on this see also http://hmgaudecker.github.io/econ-project-templates/.