# project-prog-econ
This repository contains the finale project of Tobias Werner in the class 'Effective Programming Practices for Economist'.

All code was written by Tobias Felix Werner. This includes the implementation of the bagging algorithm, the simulations/calculations, the creation of figures and the implementation to waf.

The authors of the paper are Robin Kraft, Tobias Felix Werner and David Zeimentz. It was the term paper project 'Bagging Regression Trees' in the project module
'Econometrics and Statistics' in the winter semester 2017/2018 taught by Professor Alois Kneip and Professor Dominik Liebl at the University of Bonn.

The documentation can be found under https://tofewe.github.io/bagging-documentation/index.html.


# How to run parts of the code?

The project has been implemented using Python and a replication [template](https://github.com/hmgaudecker/econ-project-templates) by Prof. Hans-Martin von Gaudecker, which builds on the build automation tool [waf](https://waf.io/).

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
Depended on your LaTeX distribution you will also need to install further LaTeX packages (like ``algorithm``), which are naturally not included in the conda environment. You will get notified about this during the first build process.

The replication template was obtained from https://github.com/hmgaudecker/econ-project-templates.
For further information on this see also http://hmgaudecker.github.io/econ-project-templates/.
