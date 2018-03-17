"""
The module which created Figure 5 of the final paper can be found under
*src.final.main_simulation.plot_simulation_convergence*. and a figure with the same
style for the indicator function for the appendix. The calculations for this
have been performed in the module *calc_simulation_convergence*, which can be found
under *src.analysis.main_simulation* and has been described in :ref:`analysis`.
The *.pickle* files, which were created by the module described above and which are
used here, where saved under *bld.out.analysis.main_simulation*.

"""
import pickle
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj


def plot_convergence(
        settings_plotting,
        convergence_settings,
        models,
        appendix):
    """
    A function that creates figure 5 in the final paper and a figure with the
    same style for the indicator function for the appendix.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared
        across various modules.

    convergence_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the
        convergence of the Bagging Algorithm.

    models: list of shape = 2
        The list of regression functions that should be contained in the figure.
        Must be of length 2. In the specification chosen in the paper, it will
        plot the Friedman 1 Model and the Linear Model.

    appendix: bool
        Indicate if we create the figure for the appendix. This simply implies
        that the figure will only contain one model.
        Therefore the figure size and the legend positioning will be adjusted
        accordingly.

    """
    plt.style.use([settings_plotting['style']])

    if appendix:
        fig = plt.figure(figsize=settings_plotting['figsize']['single_model'])
    else:
        fig = plt.figure(figsize=settings_plotting['figsize']['two_models'])

    # Create the array as it was used in the calculations (same dict and
    # method).
    n_bootstraps_array = (
        np.arange(
            convergence_settings['min_bootstrap'],
            convergence_settings['max_bootstrap'],
            convergence_settings['steps_bootstrap']
        )
    )

    # Loop over all models that we consider and add the subplot to the figure.
    for index, model in enumerate(models):
        with open(ppj("OUT_ANALYSIS_MAIN",
                      "output_convergence_{}.pickle".format(model)), "rb") as in_file:
            output_convergence = pickle.load(in_file)

        # Check if we plot the indicator model for the appendix, which would
        # mean that we only need one subplot.
        if appendix:
            axis = fig.add_subplot(1, 1, index + 1)
        else:
            axis = fig.add_subplot(1, 2, index + 1)
        mse_converged = (
            np.ones(n_bootstraps_array.size) * output_convergence['bagging_large'][0]
        )
        axis.plot(
            n_bootstraps_array,
            output_convergence['bagging_range'][:, 0],
            ls=settings_plotting['ls']['mse'],
            color=settings_plotting['colors']['bagging'],
            label='$MSPE$'
        )
        axis.plot(
            n_bootstraps_array,
            output_convergence['bagging_range'][:, 1],
            ls=settings_plotting['ls']['bias'],
            color=settings_plotting['colors']['bagging'],
            label='$Bias^{2}$'
        )
        axis.plot(
            n_bootstraps_array,
            output_convergence['bagging_range'][:, 2],
            ls=settings_plotting['ls']['variance'],
            color=settings_plotting['colors']['bagging'],
            label='$Variance$'
        )
        axis.plot(
            n_bootstraps_array,
            mse_converged,
            ls=settings_plotting['ls']['mse'],
            color=settings_plotting['colors']['converged']
        )
        axis.set_xlabel('$B$')
        axis.set_title(('$' + model.capitalize() + r' \: Model$'))

        handles_fig, labels_fig = axis.get_legend_handles_labels()

    # Adjust the positioning of the legend accordingly and save.
    if appendix:
        plt.legend(
            ncol=3, loc='lower left',
            bbox_to_anchor=(0.05, -0.27),
            frameon=True, fontsize=12,
            handles=handles_fig, labels=labels_fig
        )
        fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
        fig.savefig(
            ppj("OUT_FIGURES_MAIN", "plot_simulation_convergence_appendix.pdf"),
            bbox_inches='tight'
        )

    else:
        plt.legend(
            ncol=3, loc='lower left',
            bbox_to_anchor=(-0.47, -0.27),
            frameon=True, fontsize=12,
            handles=handles_fig, labels=labels_fig
        )
        fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
        fig.savefig(
            ppj("OUT_FIGURES_MAIN", "plot_simulation_convergence.pdf"),
            bbox_inches='tight'
        )


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS", "settings_plotting.json")) as f:
        SETTINGS_PLOTTING_IMPORTED = json.load(f)

    # For the main plots of the paper, we only use the 'friedman' and the
    # 'linear' model.
    DGP_MODELS_IN_PLOT = ['friedman', 'linear']
    DGP_MODEL_APPENDIX = ['indicator']

    # Those settings are not in the decencies for waf, as the output pickle
    # file that we load already depends on them.
    # Hence, it is redundant to specify it again as a dependency as it is
    # already implied.
    with open(ppj("IN_MODEL_SPECS", "convergence_settings.json")) as f:
        CONVERGENCE_SETTINGS_IMPORTED = json.load(f)

    plot_convergence(
        SETTINGS_PLOTTING_IMPORTED,
        CONVERGENCE_SETTINGS_IMPORTED,
        DGP_MODELS_IN_PLOT,
        appendix=False
    )
    plot_convergence(
        SETTINGS_PLOTTING_IMPORTED,
        CONVERGENCE_SETTINGS_IMPORTED,
        DGP_MODEL_APPENDIX,
        appendix=True
    )
