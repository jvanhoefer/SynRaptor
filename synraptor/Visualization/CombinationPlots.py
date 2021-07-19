"""Visualization of dose response models."""
import matplotlib.pyplot as plt
import matplotlib.axes

import itertools
import warnings

import numpy as np
from typing import List, Union

from ..CombinationModels import CombinationModelBase


def plot_3d_surface(combination_model: CombinationModelBase,
                    dose_axis_limits: List[List] = None,
                    dose_axis_scale: str = 'lin',
                    n_grid: int = 10,
                    axis_labels: List[str] = None,
                    dose_data: np.array = None,
                    response_data: np.array = None,
                    ci_alpha: float = None,
                    title: str = None,
                    model_colour: str = 'forestgreen',
                    data_colour: str = 'forestgreen',
                    ax=None):
    """
    Plots a 3d surface of a combination of two dose response models.

    Parameters:
    -----------
    combination_model:
        Combination Model, that shall be plotted.
    dose_axis_limits:
        list containing the limits of the two drug doses.
    dose_axis_scale:
        scale of the dose axis. Possible Values: 'lin', 'log'
    n_grid:
        number of grid points per axis for plotting.
        (number of predictions = n_steps^2)
    axis_labels:
        labels for the axes. Should have the either the form
        ['DrugA', 'DrugB', 'Response'] or ['DrugA', 'DrugB']
    dose_data:
        list of doses. Should have dimension (n_doses, 2)
    response_data:
        list of responses (for the doses given in dose data).
        Several replicates can be given in the same line.
    ci_alpha:
        Confidence level of the confidence interval for a validation
        experiment. If None is given, no CI is plotted.
    title:
        Title of the plot
    model_colour:
        Colour of the model prediction.
        (See https://matplotlib.org/stable/gallery/color/named_colors.html)
    data_colour:
        Colour of the dose response data.
        (See https://matplotlib.org/stable/gallery/color/named_colors.html)
    ax:
        Axes object used for plotting. Must be of type
        `matplotlib.axes._subplots.Axes3DSubplot`, in oder to be usable for
        3D plots.
    """

    if not _check_number_of_drugs(combination_model):
        raise ValueError("Combinations must contain exactly two drugs in "
                         "order to plot a 3D dose-response surface")

    # get axis limits
    if dose_axis_limits is not None:
        x_min = dose_axis_limits[0][0]
        x_max = dose_axis_limits[0][1]

        y_min = dose_axis_limits[1][0]
        y_max = dose_axis_limits[1][1]
    else:  # get default values...

        if (combination_model.drug_list[0].dose_data is None) or \
                (combination_model.drug_list[1].dose_data is None):
            raise RuntimeError("For Dose combinations without dose response "
                               "data, axis limits must be supplied.")

        x_min = np.min(combination_model.drug_list[0].dose_data)
        x_max = np.max(combination_model.drug_list[0].dose_data)

        y_min = np.min(combination_model.drug_list[1].dose_data)
        y_max = np.max(combination_model.drug_list[1].dose_data)

    x = np.linspace(x_min, x_max, n_grid)
    y = np.linspace(y_min, y_max, n_grid)

    X, Y = np.meshgrid(x, y)

    # compute responses
    Z = np.zeros_like(X)
    for (i_x, dose_x), (i_y, dose_y) in itertools.product(enumerate(x),
                                                          enumerate(y)):
        Z[i_x, i_y] = combination_model.get_combined_effect([dose_x, dose_y])

    # type check axis:
    if ax is not None and isinstance(ax, matplotlib.axes._subplots.Axes3DSubplot):
        ax = None
        warnings.warn(RuntimeWarning,
                      "Axis object is not instance of "
                      "`matplotlib.axes._subplots.Axes3DSubplot` and hence "
                      "can not be used for 3D plotting. Therefore this input "
                      "will be ignored.")
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # plot
    ax.plot_surface(X, Y, Z,
                    color=model_colour)

    # check+perform axes labels
    if axis_labels is None:
        axis_labels = ['drugA', 'drugB', 'response']
    elif len(axis_labels) == 2:
        axis_labels = [axis_labels[0], axis_labels[1], 'response']
    elif len(axis_labels) not in (2, 3):
        raise ValueError("Number of axis labels is not in (2, 3), "
                         "i.e. does not match number of axes.")

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    # format axes bounds ...
    if dose_axis_scale is 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    # plot data
    if (dose_data is not None) and (response_data is not None):
        # iterate over doses
        for idx, doses in enumerate(dose_data):
            # iterate over replicates
            for r in response_data[idx, :]:
                ax.scatter(doses[0], doses[1], r,
                           marker='*',
                           c=data_colour)

    # compute the CIs and plot them.
    if ci_alpha is not None:
        ci_min = np.zeros_like(Z)
        ci_max = np.zeros_like(Z)

        for (i_x, dose_x), (i_y, dose_y) in itertools.product(enumerate(x),
                                                              enumerate(y)):

            ci = combination_model.get_likelihood_ratio_ci([dose_x, dose_y],
                                                           alpha=ci_alpha)
            ci_min[i_x, i_y] = ci[0]
            ci_max[i_x, i_y] = ci[1]

        # plot
        ax.plot_surface(X, Y, ci_min,
                        color=data_colour, alpha=0.5)
        ax.plot_surface(X, Y, ci_max,
                        color=data_colour, alpha=0.5)

    if title is not None:
        ax.set_title(title)


def plot_3d_quantiles(combination_model: CombinationModelBase):
    raise NotImplementedError


def plot_combination_sample_histogram(combination_model: CombinationModelBase,
                                      dose_combination: List,
                                      response_data: Union[np.array, List[List], float] = None,
                                      n_samples: int = 10000,
                                      n_bins: int = 20,
                                      title: str = None,
                                      x_axis_label: str = 'dose',
                                      y_axis_label: str = 'response',
                                      colour: str = 'forestgreen'):
    """
    plots a histogram of sampled combination responses.

    Parameters
    ----------
    combination_model:
        Combination Model, that shall be plotted.
    dose_combination:
        dose combination, at which the histogram should be evaluated.
    response_data:
        response data. If multiple replicates are provided, the sigma of the
        sampling is adapted to match the sample mean of the replicates.
    n_samples:
        number of samples, that are plotted.
    n_bins:
        number of bins of the histogram.
    title:
        title of the histogram.
    x_axis_label:
        label of the x axis.
    y_axis_label:
        label of the y axis.
    colour:
        Colour of the histogram.
        (See https://matplotlib.org/stable/gallery/color/named_colors.html)

    """
    sigma = combination_model.sigma

    if response_data is None:
        n_replicates = 1
    elif isinstance(response_data, float):
        n_replicates = 1
    else:
        n_replicates = len(response_data)

    response_samples = combination_model.get_sampling_predictions(
        dose_combination=dose_combination,
        sigma=sigma,
        n_replicates=n_replicates,
        n_samples=n_samples)

    plt.hist(response_samples,
             density=True,
             color=colour,
             bins=n_bins)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if title is not None:
        plt.title(title)

    # set right and upper axis visibility to False
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def _check_number_of_drugs(combination_model: CombinationModelBase):
    """checks, if the number of drugs is two (for plotting 3d surfaces)."""
    return combination_model.n_drugs == 2

