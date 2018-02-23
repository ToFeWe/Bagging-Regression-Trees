# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:51:24 2017

@author: Tobias Werner

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

from bld.project_paths import project_paths_join as ppj



def plot_convergence_two_models(settings_plotting, convergence_settings, models):
    '''TBT X-X
    '''

    plt.style.use([settings_plotting['style']])
    fig = plt.figure(figsize=settings_plotting['figsize'])

    n_bootstraps_array = np.arange(convergence_settings['min_bootstrap'],convergence_settings['max_bootstrap'],convergence_settings['steps_bootstrap'])

    for index, model in enumerate(models):
        with open(ppj("OUT_ANALYSIS_MAIN","output_convergence_{}.pickle".format(model)), "rb") as in_file:
            output_convergence = pickle.load(in_file)


        ax = fig.add_subplot(2,2,index+1)
        mse_converged = np.ones(n_bootstraps_array.size) * output_convergence['bagging_large'][0]
        ax.plot(n_bootstraps_array,output_convergence['bagging_range'][:,0],ls=settings_plotting['ls']['mse'], color=settings_plotting['colors']['bagging'],label='$MSPE$')
        ax.plot(n_bootstraps_array,output_convergence['bagging_range'][:,1],ls=settings_plotting['ls']['bias'],color=settings_plotting['colors']['bagging'],label='$Bias^{2}$')
        ax.plot(n_bootstraps_array,output_convergence['bagging_range'][:,2],ls=settings_plotting['ls']['variance'],color=settings_plotting['colors']['bagging'],label='$Variance$')
        ax.plot(n_bootstraps_array, mse_converged,ls=settings_plotting['ls']['mse'],  color=settings_plotting['colors']['converged'])
        ax.set_xlabel('$B$')
        ax.set_title(('$'+ model.capitalize()+' \: Model$'))

    ax.legend(bbox_to_anchor=(-0.6, -0.3), ncol=3,loc='lower left',frameon=True, fontsize=15)
    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)

    fig.savefig(ppj("OUT_FIGURES_MAIN","plot_simulation_convergence.pdf"), bbox_inches='tight')

if __name__ == '__main__':

    with open(ppj("IN_MODEL_SPECS","settings_plotting.json")) as f:
        settings_plotting_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS","dgp_models.json")) as f:
        dgp_models_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS","convergence_settings.json")) as f:
        convergence_settings_imported = json.load(f)

    # We use only the first two models for plotting.
    dgp_models_in_plot =dgp_models_imported[:2]

    plot_convergence_two_models(settings_plotting_imported, convergence_settings_imported, dgp_models_in_plot)