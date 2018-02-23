# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:04:39 2017

@author: Tobias Werner
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

from bld.project_paths import project_paths_join as ppj


def plot_normal_splits(settings_plotting, normal_splits_settings, output_normal_splits):
    '''TBT X-X
    '''
    plt.style.use([settings_plotting['style']])
    fig, ax = plt.subplots(figsize=settings_plotting['figsize_theory'], ncols=3)

    print(output_normal_splits.keys())
    # Variance Subplot
    ax[0].plot(output_normal_splits['c_range'],output_normal_splits[0]['variance'], label = '$orginal$', color = settings_plotting['colors']['normal_splits'][0])
    ax[0].plot(output_normal_splits['c_range'],output_normal_splits[1]['variance'], label = r'$a=\frac{2}{3}$', color = settings_plotting['colors']['normal_splits'][1])
    ax[0].plot(output_normal_splits['c_range'],output_normal_splits[2]['variance'], label = r'$a=\frac{1}{2}$', color = settings_plotting['colors']['normal_splits'][2])
    ax[0].plot(output_normal_splits['c_range'],output_normal_splits[3]['variance'], label = r'$a=\frac{1}{10}$', color = settings_plotting['colors']['normal_splits'][3])
    ax[0].set_title('$Variance$')
    ax[0].set_xlabel('$c$')


    # Bias Subplot

    ax[1].plot(output_normal_splits['c_range'],output_normal_splits[0]['bias'], label = '$orginal$', color = settings_plotting['colors']['normal_splits'][0])
    ax[1].plot(output_normal_splits['c_range'],output_normal_splits[1]['bias'], label = r'$a=\frac{2}{3}$', color = settings_plotting['colors']['normal_splits'][1])
    ax[1].plot(output_normal_splits['c_range'],output_normal_splits[2]['bias'], label = r'$a=\frac{1}{2}$', color = settings_plotting['colors']['normal_splits'][2])
    ax[1].plot(output_normal_splits['c_range'],output_normal_splits[3]['bias'], label = r'$a=\frac{1}{10}$', color = settings_plotting['colors']['normal_splits'][3])
    ax[1].set_title('$Bias^{2}$')
    ax[1].set_xlabel('$c$')
    ax[1].legend(bbox_to_anchor=(-0.7, -0.6),ncol=4 ,loc='lower left',frameon=True, fontsize=15)



    # AMSE Subplot
    ax[2].plot(output_normal_splits['c_range'],output_normal_splits[0]['mse'], label = '$orginal$', color = settings_plotting['colors']['normal_splits'][0])
    ax[2].plot(output_normal_splits['c_range'],output_normal_splits[1]['mse'], label = r'$a=\frac{2}{3}$', color = settings_plotting['colors']['normal_splits'][1])
    ax[2].plot(output_normal_splits['c_range'],output_normal_splits[2]['mse'], label = r'$a=\frac{1}{2}$', color = settings_plotting['colors']['normal_splits'][2])
    ax[2].plot(output_normal_splits['c_range'],output_normal_splits[3]['mse'], label = r'$a=\frac{1}{10}$', color = settings_plotting['colors']['normal_splits'][3])
    ax[2].set_title('$AMSE$')
    ax[2].set_xlabel('$c$')

    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
    fig.savefig(ppj("OUT_FIGURES_THEORY","plot_normal_splits.pdf"), bbox_inches='tight')


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS","settings_plotting.json")) as f:
        settings_plotting_imported = json.load(f)

    with open(ppj("IN_MODEL_SPECS","normal_splits_settings.json")) as f:
        normal_splits_settings_imported = json.load(f)

    with open(ppj("OUT_ANALYSIS_THEORY","output_normal_splits.pickle"), "rb") as f:
        output_normal_splits_imported = pickle.load(f)

    plot_normal_splits(settings_plotting_imported, normal_splits_settings_imported, output_normal_splits_imported)
