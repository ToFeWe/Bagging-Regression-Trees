"""
A module which creates figure 6 in the final paper with the Friedman and the Linear Model and a figure with the same style for the indicator function for the appendix. The calculations for this have been performed in the module
*calc_simulation_subagging*, which can be found under *src.analysis.main_simulation* and has been described
in :ref:`analysis`.

"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import json

from bld.project_paths import project_paths_join as ppj


def plot_subagging(settings_plotting, subagging_settings, models, appendix):
    """
    A function that creates figure 6 in the final paper and a figure with the same style for the indicator function for the appendix.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared across various modules.

    subagging_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the subagging simulation.

    models: list of shape = 2
        The list of regression functions that should be contained in the figure. Must be of length 2.
        In the specification chosen in the paper, it will plot the Friedman 1 Model and the Linear Model.
        Another option is to replace one of those by the indicator model by changing the *dgp_models.json*
        described in :ref:`model_specs`.
        Note that we also need to change the *wscript* file in *src.analysis.main_simulation* then!
    appendix: bool
        Indicate if we plot a model for the appendix. This means that the figure will contain only one subplot.

    """

    plt.style.use([settings_plotting['style']])

    if appendix:
        fig = plt.figure(figsize=settings_plotting['figsize']['single_model'])
    else:
        fig = plt.figure(figsize=settings_plotting['figsize']['two_models'])


    # Create the array as it was used in the calculations (same dict and method).
    ratio_range = np.linspace(subagging_settings['min_ratio'], subagging_settings['max_ratio'] ,subagging_settings["n_ratios"])

    for index, model in enumerate(models):
        with open(ppj("OUT_ANALYSIS_MAIN","output_subagging_{}.pickle".format(model)), "rb") as in_file:
            output_subagging = pickle.load(in_file)


        # MSE for Bagging is constant.
        bagging_mse_plot = np.ones(subagging_settings["n_ratios"]) * output_subagging['bagging'][0]
        # For ratio=1 subagging is the same as fitting a single tree.
        tree_mse_plot = np.ones(subagging_settings["n_ratios"]) * output_subagging['subagging'][-1,0]

        # Check if we plot the indicator model for the appendix, which would mean that we only need one subplot.
        if appendix:
            ax = fig.add_subplot(1, 1, index+1)
        else:
            ax = fig.add_subplot(1, 2, index+1)


        ax.plot(ratio_range,output_subagging['subagging'][:,0], color=settings_plotting['colors']['subagging'],
                ls=settings_plotting['ls']['mse'], label = '$MSPE \: Subagging$')
        ax.plot(ratio_range,output_subagging['subagging'][:,1], color=settings_plotting['colors']['subagging'],
                ls=settings_plotting['ls']['bias'], label = '$Bias^{2} \: Subagging$')
        ax.plot(ratio_range,output_subagging['subagging'][:,2], color=settings_plotting['colors']['subagging'],
                ls=settings_plotting['ls']['variance'], label = '$Variance \: Subagging$')
        ax.plot(ratio_range,bagging_mse_plot,ls=settings_plotting['ls']['mse'], color=settings_plotting['colors']['bagging'], label = '$MSPE \: Bagging$')
        ax.plot(ratio_range,tree_mse_plot,ls=settings_plotting['ls']['mse'], color=settings_plotting['colors']['trees'],  label = '$MSPE \: Tree$')
        ax.set_xlabel('$a$')
        ax.set_title(('$'+ model.capitalize()+' \: Model$'))
        # Kann man dann ja noch entscheiden, ob man das folgende drin lässt
        start_y, end_y = ax.get_ylim()
        tick = 1 if end_y>5 else 0.5

        ax.yaxis.set_ticks(np.arange(0, end_y, tick))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        start_x, end_x = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(ratio_range.min(),
                                     ratio_range.max()+0.1,
                                     0.1))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        handles_fig, labels_fig = ax.get_legend_handles_labels()

    # Adjust the positioning of the legend accordingly and save.
    if appendix:
        plt.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.2, -0.27), frameon=True, fontsize=12,
                   handles=handles_fig, labels=labels_fig)
        fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
        fig.savefig(ppj("OUT_FIGURES_MAIN","plot_simulation_subagging_appendix.pdf"), bbox_inches='tight')
    else:
        plt.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.77, -0.27), frameon=True, fontsize=12,
                   handles=handles_fig, labels=labels_fig)
        fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
        fig.savefig(ppj("OUT_FIGURES_MAIN", "plot_simulation_subagging.pdf"), bbox_inches='tight')


if __name__ == '__main__':

    with open(ppj("IN_MODEL_SPECS","settings_plotting.json")) as f:
        settings_plotting_imported = json.load(f)

    # For the main plots of the paper, we only use the 'friedman' and the 'linear' model.
    dgp_models_in_plot = ['friedman', 'linear']
    dgp_model_appendix = ['indicator']

    with open(ppj("IN_MODEL_SPECS", "subagging_settings.json")) as f:
        subagging_settings_imported = json.load(f)

    plot_subagging(settings_plotting_imported, subagging_settings_imported, dgp_models_in_plot, appendix=False)
    plot_subagging(settings_plotting_imported, subagging_settings_imported, dgp_model_appendix, appendix=True)
