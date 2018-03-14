"""
A module which creates figure 2 in the final paper. The calculations for this
have been performed in the module *calc_finite_sample*, which can be found under
 *src.analysis.theory_simulation* and has been described in :ref:`analysis`.

"""
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj


def plot_finite_sample(settings_plotting, output_finite_sample):
    """
    A function that creates figure 5 in the final paper.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared
        across various modules.

    output_finite_sample: Dictionary as defined by *calc_finite_sample* in
    *src.analysis.theory_simulation*
        The dictionary that contains the simulation results for bagging the
        indicator function for different sample sizes.

    """

    plt.style.use([settings_plotting['style']])
    fig = plt.figure(figsize=settings_plotting['figsize']['finite_sample'])

    x_grid = output_finite_sample['x_range']

    # Pop x_grid as it makes the plotting easier.
    output_finite_sample.pop('x_range', None)

    # Loop over the keys (different sample sizes) to plot each.
    for index, key in enumerate(output_finite_sample.keys()):

        axis = fig.add_subplot(2, 2, index + 1)
        # Set ylim to make the effect of convergence more clear.
        axis.set_ylim([0, 0.5])
        axis.plot(
            x_grid,
            output_finite_sample[key]['mse_unbagged'],
            color=settings_plotting['colors']['trees'],
            label=r'$\hat{\theta}_{n}(x)$'
        )
        axis.plot(
            x_grid,
            output_finite_sample[key]['mse_bagging'],
            color=settings_plotting['colors']['bagging'],
            label=r'$\hat{\theta}_{n;B}(x)$'
        )
        axis.set_xlabel('$x$')
        axis.set_ylabel('$MSE$')
        axis.set_title('$n=' + str(key) + '$')

        handles_fig, labels_fig = axis.get_legend_handles_labels()

    plt.legend(
        ncol=3, loc='lower left',
        bbox_to_anchor=(-0.40, -0.27),
        frameon=True, fontsize=12,
        handles=handles_fig, labels=labels_fig
    )
    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)

    fig.savefig(
        ppj("OUT_FIGURES_THEORY", "plot_finite_sample.pdf"),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS", "settings_plotting.json")) as f:
        SETTINGS_PLOTTING_IMPORTED = json.load(f)

    with open(ppj("OUT_ANALYSIS_THEORY", "output_finite_sample.pickle"), "rb") as f:
        OUTPUT_FINITE_SAMPLE_IMPORTED = pickle.load(f)

    plot_finite_sample(
        SETTINGS_PLOTTING_IMPORTED,
        OUTPUT_FINITE_SAMPLE_IMPORTED
    )
