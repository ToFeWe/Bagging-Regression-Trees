"""
The module which created Figure 4 of the final paper can be found under
*src.final.theory_simulation.plot_normal_splits*. The calculations for this
have been performed in the module *calc_normal_splits*, which can be found under
*src.analysis.theory_simulation* and has been described in :ref:`analysis`.
The *.pickle* files, which were created by the module described above and which are
used here, where saved under *bld.out.analysis.theory_simulation*.

"""
import pickle
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bld.project_paths import project_paths_join as ppj


def plot_normal_splits(settings_plotting, settings_normal_splits, output_normal_splits):
    """
    A function that creates figure  in the final paper.

    Parameters
    ----------
    settings_plotting: Dictionary as described in :ref:`model_specs`
        The dictionary contains all plotting specifications that are shared
        across various modules.

    settings_normal_splits: Dictionary as described in :ref:`model_specs`
        The dictionary defines the calculation set-up that is specific to the
        stump predictor simulation.

    output_normal_splits: Dictionary as defined by *calc_normal_splits* in
    *src.analysis.theory_simulation*
        The dictionary that contains the simulation results for subagging of
        stump predictors for a range of subsampling fractions.

    """
    plt.style.use([settings_plotting['style']])
    fig, axis = plt.subplots(
        figsize=settings_plotting['figsize']['theory'],
        ncols=3
    )

    # Create the Variance Subplot with index 0.
    axis[0].plot(
        output_normal_splits['c_range'],
        output_normal_splits[0]['variance'],
        label='$orginal$',
        color=settings_plotting['colors']['normal_splits'][0]
    )
    axis[0].plot(
        output_normal_splits['c_range'],
        output_normal_splits[1]['variance'],
        color=settings_plotting['colors']['normal_splits'][1]
    )
    axis[0].plot(
        output_normal_splits['c_range'],
        output_normal_splits[2]['variance'],
        color=settings_plotting['colors']['normal_splits'][2]
    )
    axis[0].plot(
        output_normal_splits['c_range'],
        output_normal_splits[3]['variance'],
        color=settings_plotting['colors']['normal_splits'][3]
    )
    axis[0].set_title('$Variance$')
    axis[0].set_xlabel('$c$')

    # Create the Bias Subplot with index 1.
    axis[1].plot(
        output_normal_splits['c_range'],
        output_normal_splits[0]['bias'],
        color=settings_plotting['colors']['normal_splits'][0]
    )
    axis[1].plot(
        output_normal_splits['c_range'],
        output_normal_splits[1]['bias'],
        color=settings_plotting['colors']['normal_splits'][1]
    )
    axis[1].plot(
        output_normal_splits['c_range'],
        output_normal_splits[2]['bias'],
        color=settings_plotting['colors']['normal_splits'][2]
    )
    axis[1].plot(
        output_normal_splits['c_range'],
        output_normal_splits[3]['bias'],
        color=settings_plotting['colors']['normal_splits'][3]
    )
    axis[1].set_title('$Bias^{2}$')
    axis[1].set_xlabel('$c$')

    # Create the AMSE Subplot with index 2.
    axis[2].plot(
        output_normal_splits['c_range'],
        output_normal_splits[0]['mse'],
        label='$orginal$',
        color=settings_plotting['colors']['normal_splits'][0]
    )
    axis[2].plot(
        output_normal_splits['c_range'],
        output_normal_splits[1]['mse'],
        label=(
            r'$a=\frac{{{num}}}{{{denum}}}$'.format(
                num=settings_normal_splits['a_array']['second_a'][0],
                denum=settings_normal_splits['a_array']['second_a'][1]
            )
        ),
        color=settings_plotting['colors']['normal_splits'][1]
    )
    axis[2].plot(
        output_normal_splits['c_range'],
        output_normal_splits[2]['mse'],
        label=(
            r'$a=\frac{{{num}}}{{{denum}}}$'.format(
                num=settings_normal_splits['a_array']['third_a'][0],
                denum=settings_normal_splits['a_array']['third_a'][1]
            )
        ),
        color=settings_plotting['colors']['normal_splits'][2]
    )
    axis[2].plot(
        output_normal_splits['c_range'],
        output_normal_splits[3]['mse'],
        label=(
            r'$a=\frac{{{num}}}{{{denum}}}$'.format(
                num=settings_normal_splits['a_array']['fourth_a'][0],
                denum=settings_normal_splits['a_array']['fourth_a'][1]
            )
        ),
        color=settings_plotting['colors']['normal_splits'][3]
    )
    axis[2].set_title('$AMSE$')
    axis[2].set_xlabel('$c$')
    handles_fig, labels_fig = axis[2].get_legend_handles_labels()

    plt.legend(
        ncol=4, loc='lower left',
        bbox_to_anchor=(-1.55, -0.4),
        frameon=True, fontsize=12,
        handles=handles_fig, labels=labels_fig
    )

    fig.tight_layout(pad=0.4, w_pad=1, h_pad=2.5)
    fig.savefig(
        ppj("OUT_FIGURES_THEORY", "plot_normal_splits.pdf"),
        bbox_inches='tight'
    )


if __name__ == '__main__':
    with open(ppj("IN_MODEL_SPECS", "settings_plotting.json")) as f:
        SETTINGS_PLOTTING_IMPORTED = json.load(f)

    with open(ppj("IN_MODEL_SPECS", "normal_splits_settings.json")) as f:
        NORMAL_SPLITS_SETTINGS_IMPORTED = json.load(f)

    with open(ppj("OUT_ANALYSIS_THEORY", "output_normal_splits.pickle"), "rb") as f:
        OUTPUT_NORMAL_SPLITS_IMPORTED = pickle.load(f)

    plot_normal_splits(
        SETTINGS_PLOTTING_IMPORTED,
        NORMAL_SPLITS_SETTINGS_IMPORTED,
        OUTPUT_NORMAL_SPLITS_IMPORTED
    )
