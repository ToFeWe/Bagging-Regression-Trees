# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:16:00 2017

@author: Tobias Werner
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

from bld.project_paths import project_paths_join as ppj

def plot_finite_sample(settings_plotting, output_finite_sample):
    '''TBT X-X
    '''

    plt.style.use([settings_plotting['style']])
    fig = plt.figure(figsize=settings_plotting['figsize'])

    x_grid = output_finite_sample['x_range']

    # Pop x_grid as it makes the plotting easiert for me
    output_finite_sample.pop('x_range',None)

    for index, key in enumerate(output_finite_sample.keys()):

        ax = fig.add_subplot(2,2,index+1)
        ax.set_ylim([0,0.5])
        ax.plot(x_grid,output_finite_sample[key]['mse_unbagged'], color=settings_plotting['colors']['trees'],
                label=r'$\hat{\theta}_{n}(x)$')
        ax.plot(x_grid,output_finite_sample[key]['mse_bagging'], color=settings_plotting['colors']['bagging'],
                label=r'$\hat{\theta}_{n;B}(x)$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$MSE$')
        ax.set_title('$n='+str(key)+'$')

    ax.legend(bbox_to_anchor=(-0.45, -0.3), ncol=3,loc='lower left',frameon=True, fontsize=15)
    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)

    fig.savefig(ppj("OUT_FIGURES_THEORY","plot_finite_sample.pdf"), bbox_inches='tight')


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS","settings_plotting.json")) as f:
        settings_plotting_imported = json.load(f)

    with open(ppj("OUT_ANALYSIS_THEORY","output_finite_sample.pickle"), "rb") as f:
        output_finite_sample_imported = pickle.load(f)

    plot_finite_sample(settings_plotting_imported, output_finite_sample_imported)
