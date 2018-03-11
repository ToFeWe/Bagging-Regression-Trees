.. _overview:

**************************************
The conceptual overview of the project
**************************************

Overview for the project
========================
The implementation of the simulations described in the term paper has been split up across various modules. The objective of this section is
to explain the conception and structure.

The part :ref:`model_code` contains explanations for all basic algorithms and functions that are used for the main parts of the simulation, which can be found in Section 5. Amongst them is the implementation
of the Bagging Algorithm for Regression Trees and the ``MonteCarloSimulation`` Class, which is used to perform the simulations of Section 5 of the term paper.
Within :ref:`analysis` the documentation for the main calculations and simulations of the paper can be found. The analysis part has been split up into three subfolders to maintain a clear
structure for the project. The three subfolders coincide with the three computationally intensive parts of the paper, which are the calculations for the theoretical part in Sections 3 and 4,
the main simulations in Section 5 and the application of the Bagging Algorithm to real data in Section 6. As the simulation setup differs drastically between those parts, it made sense
to reflect this also in the structure of the project.
In :ref:`final` the modules for creating the figures and tables are documented. Note that here we also maintain the structure chosen in :ref:`analysis`.
All model specifications for the different simulations are defined in .json files.This way we can easily ensure that modules/simulations that should share common attributes
actually share them. An overview for the different .json files can be found in :ref:`model_specs`.
Finally, the overview of the final output files is listed in :ref:`paper`. In this folder you can also find further figures, that have not been created during the
build process.

Overview and explanations for different design choices
======================================================

In the following we will give an overview over different design choices that have been made within this project.
We will focus on those that might seem unintuitive at first glance and give a short explanation to those.




How to run parts of the code? How to replicate the output myself?
==========================================================

The term paper project has been implemented using Python and a replication template by Professor von Gaudecker, which builds on the build automation tool waf.

To run the code yourself you have to follow the following steps:
  * Make sure to have Miniconda or Anaconda installed. A modern LaTeX distribution (e.g. TeXLive, MacTex, or MikTex) needs to be found on your path.
  * Download the project or clone it to your machine.
  * Move to the root directory of the project in the shell
  * Run ``source set-env.sh`` (Linux, MacOS) or ``set-env.bat`` to set up the conda environment specific to this project.
  * Run ``python waf.py configure`` and then ``python waf.py build`` to run the whole project. Note that if you use the
    parameter specification used in the paper, this may take some time (~ 40 min with i5-6200 CPU @ 2.3 GHz, 8 GB RAM).
    If you are interested in the implementation you can also scale down the parameters of the simulations in the
    respective JSON files described in :ref:`model_specs`.

The replication template was obtained from https://github.com/hmgaudecker/econ-project-templates.
For further information on this see also http://hmgaudecker.github.io/econ-project-templates/.
