# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:42:26 2017

@author: Tobias Werner
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj
import json

def plot_toy_example(settings_plotting, output_toy_example):
    '''TBT X-X
    '''
    plt.style.use([settings_plotting['style']])
    fig, ax = plt.subplots(figsize=settings_plotting['figsize_theory'], ncols=3)

    # Variance Subplot
    ax[0].plot(output_toy_example['c_range'],output_toy_example['bagged']['variance'], color=settings_plotting['colors']['bagging'])
    ax[0].plot(output_toy_example['c_range'],output_toy_example['unbagged']['variance'], color=settings_plotting['colors']['trees'])
    ax[0].xaxis.set_ticks(np.arange(-4,4+1,2))
    ax[0].set_title('$Variance$')
    ax[0].set_xlabel('$c$')

    # Bias Subplot

    ax[1].plot(output_toy_example['c_range'], output_toy_example['bagged']['bias'],
               label = r'$\hat{\theta}_{n;B}(x_{n}(c))$',
               color=settings_plotting['colors']['bagging']
    )
    ax[1].plot(output_toy_example['c_range'],output_toy_example['unbagged']['bias'],
               label = r'$\hat{\theta}_{n}(x_{n}(c))$',
               color=settings_plotting['colors']['trees']
    )
    ax[1].xaxis.set_ticks(np.arange(-4,4+1,2))
    ax[1].set_title('$Bias^{2}$')
    ax[1].set_xlabel('$c$')
    ax[1].legend(bbox_to_anchor=(-0.25, -0.6),ncol=2 ,loc='lower left',frameon=True, fontsize=15)



    # AMSE Subplot
    ax[2].plot(output_toy_example['c_range'],
      			 np.add(output_toy_example['bagged']['bias'],
                 		 output_toy_example['bagged']['variance']),
               color=settings_plotting['colors']['bagging']
    )
    # Keep in mind that the unbagged predcitor is unbiased.
    ax[2].plot(output_toy_example['c_range'],output_toy_example['unbagged']['variance'], color=settings_plotting['colors']['trees'])
    ax[2].xaxis.set_ticks(np.arange(-4,4+1,2))
    ax[2].set_title('$AMSE$')
    ax[2].set_xlabel('$c$')

    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
    fig.savefig(ppj("OUT_FIGURES_THEORY","plot_toy_example.pdf"), bbox_inches='tight')


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS","settings_plotting.json")) as f:
        settings_plotting_imported = json.load(f)

    with open(ppj("OUT_ANALYSIS_THEORY","output_toy_example.pickle"), "rb") as f:
        output_toy_example_imported = pickle.load(f)

    plot_toy_example(settings_plotting_imported, output_toy_example_imported)
