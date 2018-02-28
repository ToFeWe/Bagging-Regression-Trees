"""
A module which creates figure 7 in the final paper. The calculations for this have been performed in the module
*calc_simulation_tree_depth*, which can be found under *src.analysis.main_simulation* and has been described
in :ref:`analysis`.

"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

from bld.project_paths import project_paths_join as ppj


def plot_tree_depth_two_models(settings_plotting, tree_depth_settings, models):
    """
    A function that creates figure 6 in the final paper.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared across various modules.

    tree_depth_settings: Dictionary as described in :ref:`model_specs`
        The dictionary defines the simulation set-up that is specific to the tree depth simulation.

    models: list of shape = 2
        The list of regression functions that should be contained in the figure. Must be of length 2.
        In the specification chosen in the paper, it will plot the Friedman 1 Model and the Linear Model.
        Another option is to replace one of those by the indicator model by changing the *dgp_models.json*
        described in :ref:`model_specs`.
        Note that we also need to change the *wscript* file in *src.analysis.main_simulation* then!

    """
    plt.style.use([settings_plotting['style']])
    fig = plt.figure(figsize=settings_plotting['figsize'])

    # Create array as it was used in the simulation.
    min_split_array = np.arange(tree_depth_settings['min_split'],tree_depth_settings['max_split']+tree_depth_settings["steps_split"],tree_depth_settings["steps_split"])[::-1]



    for index, model in enumerate(models):
        with open(ppj("OUT_ANALYSIS_MAIN","output_tree_depth_{}.pickle".format(model)), "rb") as in_file:
            output_tree_depth = pickle.load(in_file)

        ax = fig.add_subplot(2,2,index+1)

        # First plot the results for the MSE.
        ax.plot(min_split_array,output_tree_depth['bagging'][:,0],ls=settings_plotting['ls']['mse'],color=settings_plotting['colors']['bagging'],label='$MSPE \: Bagging$')
        ax.plot(min_split_array,output_tree_depth['trees'][:,0],ls=settings_plotting['ls']['mse'],color=settings_plotting['colors']['trees'],label='$MSPE \: Tree$')

        # Then the results for the squared-bias.
        ax.plot(min_split_array,output_tree_depth['bagging'][:,1],ls=settings_plotting['ls']['bias'],color=settings_plotting['colors']['bagging'],label='$Bias^{2} \: Bagging$')
        ax.plot(min_split_array,output_tree_depth['trees'][:,1],ls=settings_plotting['ls']['bias'],color=settings_plotting['colors']['trees'],label='$Bias^{2} \: Tree$')

        # And lastly for the variance.
        ax.plot(min_split_array,output_tree_depth['bagging'][:,2],ls=settings_plotting['ls']['variance'],color=settings_plotting['colors']['bagging'],label='$Variance \: Bagging$')
        ax.plot(min_split_array,output_tree_depth['trees'][:,2],ls=settings_plotting['ls']['variance'],color=settings_plotting['colors']['trees'],label='$Variance \: Tree$')

        ax.set_xlabel('$Minimal \: Size \: for \: Each \: Terminal \: Node$')
        ax.set_title(('$'+ model.capitalize()+' \: Model$'))



    ax.legend(bbox_to_anchor=(-0.87, -0.4), ncol=3,loc='lower left',frameon=True, fontsize=15)
    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)

    fig.savefig(ppj("OUT_FIGURES_MAIN","plot_simulation_tree_depth.pdf"), bbox_inches='tight')


if __name__ == '__main__':

    with open(ppj("IN_MODEL_SPECS","settings_plotting.json")) as f:
        settings_plotting_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS","dgp_models.json")) as f:
        dgp_models_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS","tree_depth_settings.json")) as f:
        tree_depth_settings_imported = json.load(f)

    # We use only the first two models for plotting.
    dgp_models_in_plot = dgp_models_imported[:2]

    plot_tree_depth_two_models(settings_plotting_imported, tree_depth_settings_imported, dgp_models_in_plot)
