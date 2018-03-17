"""
The module which created Figure 2 of the final paper can be found under
*src.final.theory_simulation.plot_toy_example*. The calculations for this
have been performed in the module *calc_toy_example*, which can be found under
*src.analysis.theory_simulation* and has been described in :ref:`analysis`.
The *.pickle* files, which were created by the module described above and which are
used here, where saved under *bld.out.analysis.theory_simulation*.

"""
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj


def plot_toy_example(settings_plotting, output_toy_example):
    """
    A function that creates figure 3 in the final paper.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared
        across various modules.

    output_toy_example: Dictionary as defined by *calc_toy_example* in
    *src.analysis.theory_simulation*
        The dictionary that contains the calculation results for the bagged and
        unbagged indicator function.

    """

    plt.style.use([settings_plotting['style']])
    fig, axis = (
        plt.subplots(
            figsize=settings_plotting['figsize']['theory'],
            ncols=3
        )
    )

    # Create the Variance Subplot with index 0.
    axis[0].plot(
        output_toy_example['c_range'],
        output_toy_example['bagged']['variance'],
        color=settings_plotting['colors']['bagging']
    )
    axis[0].plot(
        output_toy_example['c_range'],
        output_toy_example['unbagged']['variance'],
        color=settings_plotting['colors']['trees']
    )
    # Set the x-axis ticks to make it more readable.
    axis[0].xaxis.set_ticks(np.arange(-4, 4 + 1, 2))
    axis[0].set_title('$Variance$')
    axis[0].set_xlabel('$c$')

    # Create the Bias Subplot with index 1.
    axis[1].plot(
        output_toy_example['c_range'],
        output_toy_example['bagged']['bias'],
        label=r'$\hat{\theta}_{n;B}(x_{n}(c))$',
        color=settings_plotting['colors']['bagging']
    )
    axis[1].plot(
        output_toy_example['c_range'],
        output_toy_example['unbagged']['bias'],
        label=r'$\hat{\theta}_{n}(x_{n}(c))$',
        color=settings_plotting['colors']['trees']
    )
    # Set the x-axis ticks to make it more readable.
    axis[1].xaxis.set_ticks(np.arange(-4, 4 + 1, 2))
    axis[1].set_title('$Bias^{2}$')
    axis[1].set_xlabel('$c$')

    handles_fig, labels_fig = axis[1].get_legend_handles_labels()

    # AMSE Subplot
    amse_bagging = (
        np.add(
            output_toy_example['bagged']['bias'],
            output_toy_example['bagged']['variance']
        )
    )
    axis[2].plot(
        output_toy_example['c_range'],
        amse_bagging,
        color=settings_plotting['colors']['bagging']
    )
    # Keep in mind that the unbagged predictor is unbiased.
    axis[2].plot(
        output_toy_example['c_range'],
        output_toy_example['unbagged']['variance'],
        color=settings_plotting['colors']['trees']
    )
    # Set the x-axis ticks to make it more readable
    axis[2].xaxis.set_ticks(np.arange(-4, 4 + 1, 2))
    axis[2].set_title('$AMSE$')
    axis[2].set_xlabel('$c$')

    plt.legend(
        ncol=4, loc='lower left',
        bbox_to_anchor=(-1.25, -0.4),
        frameon=True, fontsize=12,
        handles=handles_fig, labels=labels_fig
    )

    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
    fig.savefig(
        ppj("OUT_FIGURES_THEORY", "plot_toy_example.pdf"),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS", "settings_plotting.json")) as f:
        SETTINGS_PLOTTING_IMPORTED = json.load(f)

    with open(ppj("OUT_ANALYSIS_THEORY", "output_toy_example.pickle"), "rb") as f:
        OUTPUT_TOY_EXAMPLE_IMPORTED = pickle.load(f)

    plot_toy_example(SETTINGS_PLOTTING_IMPORTED, OUTPUT_TOY_EXAMPLE_IMPORTED)
