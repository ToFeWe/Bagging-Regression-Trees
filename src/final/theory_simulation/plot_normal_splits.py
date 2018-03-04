"""
A module which creates figure 4 in the final paper. The calculations for this have been performed in the module
*calc_normal_splits*, which can be found under *src.analysis.theory_simulation* and has been described
in :ref:`analysis`.

"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

from bld.project_paths import project_paths_join as ppj


def plot_normal_splits(settings_plotting, output_normal_splits):
    """
    A function that creates figure  in the final paper.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared across various modules.

    output_normal_splits: Dictionary as defined by *calc_normal_splits* in *src.analysis.theory_simulation*
        The dictionary that contains the simulation results for subagging of stump predictors for a range of
        subsampling fractions.

    """
    plt.style.use([settings_plotting['style']])
    fig, ax = plt.subplots(figsize=settings_plotting['figsize']['theory'], ncols=3)

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

    plot_normal_splits(settings_plotting_imported, output_normal_splits_imported)
