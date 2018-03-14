"""
A module which creates figure 7  in the final paper and a figure with the same
style for the indicator function for the appendix. The calculations for this have
been performed in the module *calc_simulation_tree_depth*, which can be found
under *src.analysis.main_simulation* and has been described
in :ref:`analysis`.

"""
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bld.project_paths import project_paths_join as ppj


def plot_tree_depth(settings_plotting, tree_depth_settings, models, appendix):
    """
    A function that creates figure 7 in the final paper and a figure with the
    same style for the indicator function for the appendix.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared
        across various modules.

    tree_depth_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the
        tree depth simulation.

    models: list of shape = 2
        The list of regression functions that should be contained in the figure.
        Must be of length 2.
        In the specification chosen in the paper, it will plot the
        Friedman 1 Model and the Linear Model.

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
    min_split_array = (
        np.arange(
            tree_depth_settings['min_split'],
            tree_depth_settings['max_split'] +
            tree_depth_settings["steps_split"],
            tree_depth_settings["steps_split"]
        )[::-1]
    )

    for index, model in enumerate(models):
        with open(ppj("OUT_ANALYSIS_MAIN",
                      "output_tree_depth_{}.pickle".format(model)), "rb") as in_file:
            output_tree_depth = pickle.load(in_file)

        # Check if we plot the indicator model for the appendix, which would
        # mean that we only need one subplot.
        if appendix:
            axis = fig.add_subplot(1, 1, index + 1)
        else:
            axis = fig.add_subplot(1, 2, index + 1)

        # First plot the results for the MSE.
        axis.plot(
            min_split_array,
            output_tree_depth['bagging'][:, 0],
            ls=settings_plotting['ls']['mse'],
            color=settings_plotting['colors']['bagging'],
            label=r'$MSPE \: Bagging$'
        )
        axis.plot(
            min_split_array,
            output_tree_depth['trees'][:, 0],
            ls=settings_plotting['ls']['mse'],
            color=settings_plotting['colors']['trees'],
            label=r'$MSPE \: Tree$'
        )

        # Plot the results for the squared-bias.
        axis.plot(
            min_split_array,
            output_tree_depth['bagging'][:, 1],
            ls=settings_plotting['ls']['bias'],
            color=settings_plotting['colors']['bagging'],
            label=r'$Bias^{2} \: Bagging$'
        )
        axis.plot(
            min_split_array,
            output_tree_depth['trees'][:, 1],
            ls=settings_plotting['ls']['bias'],
            color=settings_plotting['colors']['trees'],
            label=r'$Bias^{2} \: Tree$'
        )

        # Plot the results for the variance.
        axis.plot(
            min_split_array,
            output_tree_depth['bagging'][:, 2],
            ls=settings_plotting['ls']['variance'],
            color=settings_plotting['colors']['bagging'],
            label=r'$Variance \: Bagging$'
        )
        axis.plot(
            min_split_array,
            output_tree_depth['trees'][:, 2],
            ls=settings_plotting['ls']['variance'],
            color=settings_plotting['colors']['trees'],
            label=r'$Variance \: Tree$'
        )

        axis.set_xlabel(r'$Minimal \: Size \: for \: Each \: Terminal \: Node$')
        axis.set_title(('$' + model.capitalize() + r' \: Model$'))

        handles_fig, labels_fig = axis.get_legend_handles_labels()

    # Adjust the positioning of the legend accordingly and save.
    if appendix:
        plt.legend(
            ncol=3, loc='lower left',
            bbox_to_anchor=(-0.15, -0.27),
            frameon=True, fontsize=12,
            handles=handles_fig, labels=labels_fig
        )
        fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
        fig.savefig(
            ppj("OUT_FIGURES_MAIN", "plot_simulation_tree_depth_appendix.pdf"),
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
            ppj("OUT_FIGURES_MAIN", "plot_simulation_tree_depth.pdf"),
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
    with open(ppj("IN_MODEL_SPECS", "tree_depth_settings.json")) as f:
        TREE_DEPTH_SETTINGS_IMPORTED = json.load(f)

    plot_tree_depth(
        SETTINGS_PLOTTING_IMPORTED,
        TREE_DEPTH_SETTINGS_IMPORTED,
        DGP_MODELS_IN_PLOT,
        appendix=False)
    plot_tree_depth(
        SETTINGS_PLOTTING_IMPORTED,
        TREE_DEPTH_SETTINGS_IMPORTED,
        DGP_MODEL_APPENDIX,
        appendix=True)
