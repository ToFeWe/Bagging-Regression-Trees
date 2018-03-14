"""
A module which creates figure 8 in the final paper. The calculations for this
have been performed in the module *calc_simulation_convergence*, which can be
found under *src.analysis.real_data_simulation* and has been described
in :ref:`analysis`.

"""
import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj


def plot_boston(settings_plotting, subagging_settings, output_boston):
    """
    A function that creates figure 8 in the final paper.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared
        across various modules.

    subagging_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the
        subagging simulation.

    output_boston: Dictionary as defined by *calc_boston* in
    *src.analysis.real_data_simulation*
        The dictionary that contains the simulation results for bagging and
        subagging (for the ratio range) for the boston housing data.

    """
    plt.style.use([settings_plotting['style']])
    fig = plt.figure(figsize=settings_plotting['figsize']['single_model'])

    # Create a range of subsampeling ratios as it was used in all simulations.
    ratio_range = (
        np.linspace(
            subagging_settings['min_ratio'],
            subagging_settings['max_ratio'],
            subagging_settings["n_ratios"]
        )
    )

    # MSE for Bagging is constant.
    bagging_mse_plot = (
        np.ones(subagging_settings["n_ratios"]) * output_boston['mse_bagging']
    )
    # For ratio=1 subagging is the same as fitting a single tree.
    tree_mse_plot = (
        np.ones(subagging_settings["n_ratios"]) * output_boston['mse_subagging'][-1]
    )

    plt.plot(
        ratio_range,
        output_boston['mse_subagging'],
        color=settings_plotting['colors']['subagging'],
        label=r'$MSPE \: Subagging$'
    )
    plt.plot(
        ratio_range,
        bagging_mse_plot,
        color=settings_plotting['colors']['bagging'],
        label=r'$MSPE \: Bagging$'
    )
    plt.plot(
        ratio_range,
        tree_mse_plot,
        color=settings_plotting['colors']['trees'],
        label=r'$MSPE \: Tree$'
    )
    plt.xlabel('$a$')
    plt.ylim(ymin=0)
    plt.title(r'$MSPE \: for \: Boston \: Housing \:Data$')
    plt.legend(
        ncol=3, loc='lower left',
        bbox_to_anchor=(-0.1, -0.27),
        frameon=True, fontsize=12
    )
    plt.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
    fig.savefig(
        ppj("OUT_FIGURES_REAL_DATA", "plot_boston.pdf"),
        bbox_inches='tight'
    )


if __name__ == '__main__':

    with open(ppj("IN_MODEL_SPECS", "settings_plotting.json")) as f:
        SETTINGS_PLOTTING_IMPORTED = json.load(f)

    # Those settings are not in the decencies for waf, as the output pickle
    # file that we load already depends on them.
    # Hence, it is redundant to specify it again as a dependency as it is
    # already implied.
    with open(ppj("IN_MODEL_SPECS", "subagging_settings.json")) as f:
        SUBAGGING_SETTINGS_IMPORTED = json.load(f)

    with open(ppj("OUT_ANALYSIS_REAL_DATA", "output_boston.pickle"), "rb") as in_file:
        OUTPUT_BOSTON_IMPORTED = pickle.load(in_file)

    plot_boston(
        SETTINGS_PLOTTING_IMPORTED,
        SUBAGGING_SETTINGS_IMPORTED,
        OUTPUT_BOSTON_IMPORTED)
