# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:29:29 2017

@author: Tobias Werner

Thos module plots the output of the simulations for a maximum of two data generating
processes. This is restricted to two due to space limations in the paper. By default
the first two entries/models within *dgp_models.json* will be plotted, in case you
decided also to simulate the indicator model.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import json

from bld.project_paths import project_paths_join as ppj

def plot_subagging_two_models(settings_plotting, subagging_settings, models):
    '''TBT X-X
    '''

    plt.style.use([settings_plotting['style']])
    fig = plt.figure(figsize=settings_plotting['figsize'])

    # Create a range of subsampeling ratios as it was used in all simulations.
    ratio_range = np.linspace(subagging_settings['min_ratio'], subagging_settings['max_ratio'] ,subagging_settings["n_ratios"])

    for index, model in enumerate(models):
        with open(ppj("OUT_ANALYSIS_MAIN","output_subagging_{}.pickle".format(model)), "rb") as in_file:
            output_subagging = pickle.load(in_file)


        # MSE for Bagging is constant.
        bagging_mse_plot = np.ones(subagging_settings["n_ratios"]) * output_subagging['bagging'][0]
        # For ratio=1 subagging is the same as fitting a single tree.
        tree_mse_plot = np.ones(subagging_settings["n_ratios"]) * output_subagging['subagging'][-1,0]

        ax = fig.add_subplot(2,2,index+1)
        ax.plot(ratio_range,output_subagging['subagging'][:,0], color=settings_plotting['colors']['subagging'],
                ls=settings_plotting['ls']['mse'], label = '$MSE \: Subagging$')
        ax.plot(ratio_range,output_subagging['subagging'][:,1], color=settings_plotting['colors']['subagging'],
                ls=settings_plotting['ls']['bias'], label = '$Bias^{2} \: Subagging$')
        ax.plot(ratio_range,output_subagging['subagging'][:,2], color=settings_plotting['colors']['subagging'],
                ls=settings_plotting['ls']['variance'], label = '$Variance \: Subagging$')
        ax.plot(ratio_range,bagging_mse_plot,ls=settings_plotting['ls']['mse'], color=settings_plotting['colors']['bagging'], label = '$MSE \: Bagging$')
        ax.plot(ratio_range,tree_mse_plot,ls=settings_plotting['ls']['mse'], color=settings_plotting['colors']['trees'],  label = '$MSE \: Tree$')
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




    ax.legend(bbox_to_anchor=(-0.93, -0.4), ncol=3,loc='lower left',frameon=True, fontsize=15)
    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
    fig.savefig(ppj("OUT_FIGURES_MAIN","plot_simulation_subagging.pdf"), bbox_inches='tight')

if __name__ == '__main__':

    with open(ppj("IN_MODEL_SPECS","settings_plotting.json")) as f:
        settings_plotting_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS","dgp_models.json")) as f:
        dgp_models_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS","subagging_settings.json")) as f:
        subagging_settings_imported = json.load(f)

    # We use only the first two models for plotting.
    dgp_models_in_plot =dgp_models_imported[:2]

    plot_subagging_two_models(settings_plotting_imported, subagging_settings_imported, dgp_models_in_plot)
