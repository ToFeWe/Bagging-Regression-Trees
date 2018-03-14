"""
A module which creates figure 6 in the final paper with the Friedman and the
Linear Model and a figure with the same style for the indicator function for the
appendix. The calculations for this have been performed in the module
*calc_simulation_subagging*, which can be found under
*src.analysis.main_simulation* and has been described in :ref:`analysis`.

"""
import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj


def plot_subagging(settings_plotting, subagging_settings, models, appendix):
    """
    A function that creates figure 6 in the final paper and a figure with the
    same style for the indicator function for the appendix.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared
        across various modules.

    subagging_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the
        subagging simulation.

    models: list of shape = 2
        The list of regression functions that should be contained in the figure.
        Must be of length 2. In the specification chosen in the paper, it will
        plot the Friedman 1 Model and the Linear Model.

    appendix: bool
        Indicate if we create the figure for the appendix. This simply implies
        that the figure will only contain one model.
        Therefore the figure size and the legend positioning will be adjusted
        accordingly.    """

    plt.style.use([settings_plotting['style']])

    if appendix:
        fig = plt.figure(figsize=settings_plotting['figsize']['single_model'])
    else:
        fig = plt.figure(figsize=settings_plotting['figsize']['two_models'])

    # Create the array as it was used in the calculations (same dict and
    # method).
    ratio_range = np.linspace(
        subagging_settings['min_ratio'],
        subagging_settings['max_ratio'],
        subagging_settings["n_ratios"]
    )

    for index, model in enumerate(models):
        with open(ppj("OUT_ANALYSIS_MAIN",
                      "output_subagging_{}.pickle".format(model)), "rb") as in_file:
            output_subagging = pickle.load(in_file)

        # MSE for Bagging is constant.
        bagging_mse_plot = (
            np.ones(subagging_settings["n_ratios"]) * output_subagging['bagging'][0]
        )
        # For ratio=1 subagging is the same as fitting a single tree.
        tree_mse_plot = (
            np.ones(subagging_settings["n_ratios"]) * output_subagging['subagging'][-1, 0]
        )

        # Check if we plot the indicator model for the appendix, which would
        # mean that we only need one subplot.
        if appendix:
            axis = fig.add_subplot(1, 1, index + 1)
        else:
            axis = fig.add_subplot(1, 2, index + 1)

        axis.plot(
            ratio_range, output_subagging['subagging'][:, 0],
            color=settings_plotting['colors']['subagging'],
            ls=settings_plotting['ls']['mse'],
            label=r'$MSPE \: Subagging$'
        )
        axis.plot(
            ratio_range, output_subagging['subagging'][:, 1],
            color=settings_plotting['colors']['subagging'],
            ls=settings_plotting['ls']['bias'],
            label=r'$Bias^{2} \: Subagging$'
        )
        axis.plot(
            ratio_range,
            output_subagging['subagging'][:, 2],
            color=settings_plotting['colors']['subagging'],
            ls=settings_plotting['ls']['variance'],
            label=r'$Variance \: Subagging$'
        )
        axis.plot(
            ratio_range,
            bagging_mse_plot,
            ls=settings_plotting['ls']['mse'],
            color=settings_plotting['colors']['bagging'],
            label=r'$MSPE \: Bagging$'
        )
        axis.plot(
            ratio_range,
            tree_mse_plot,
            ls=settings_plotting['ls']['mse'],
            color=settings_plotting['colors']['trees'],
            label=r'$MSPE \: Tree$'
        )
        axis.set_xlabel('$a$')
        axis.set_title(('$' + model.capitalize() + r' \: Model$'))

        handles_fig, labels_fig = axis.get_legend_handles_labels()

    # Adjust the positioning of the legend accordingly and save.
    if appendix:
        plt.legend(
            ncol=3, loc='lower left',
            bbox_to_anchor=(-0.2, -0.27),
            frameon=True, fontsize=12,
            handles=handles_fig, labels=labels_fig
        )
        fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
        fig.savefig(
            ppj("OUT_FIGURES_MAIN", "plot_simulation_subagging_appendix.pdf"),
            bbox_inches='tight'
        )
    else:
        plt.legend(
            ncol=3, loc='lower left',
            bbox_to_anchor=(-0.77, -0.27),
            frameon=True, fontsize=12,
            handles=handles_fig, labels=labels_fig
        )
        fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
        fig.savefig(
            ppj("OUT_FIGURES_MAIN", "plot_simulation_subagging.pdf"),
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
    with open(ppj("IN_MODEL_SPECS", "subagging_settings.json")) as f:
        SUBAGGING_SETTINGS_IMPORTED = json.load(f)

    plot_subagging(
        SETTINGS_PLOTTING_IMPORTED,
        SUBAGGING_SETTINGS_IMPORTED,
        DGP_MODELS_IN_PLOT,
        appendix=False)
    plot_subagging(
        SETTINGS_PLOTTING_IMPORTED,
        SUBAGGING_SETTINGS_IMPORTED,
        DGP_MODEL_APPENDIX,
        appendix=True)
