"""Visualization of dose response models."""

import matplotlib.pyplot as plt
import numpy as np
import pypesto.visualize as visualize

from ..DoseResponseModels import DoseResponseModelBase


def plot_dose_response_model(dose_response_model: DoseResponseModelBase,
                             dose_min: float = None,
                             dose_max: float = None,
                             model_colour: str = 'forestgreen',
                             data_colour: str = 'forestgreen',
                             title: str = None,
                             x_axis_label: str = 'dose',
                             y_axis_label: str = 'response',
                             dose_axis_scale: str = 'lin'):
    """
    Generates a plot of a dose response model.

    Parameters:
    -----------
    dose_response_model;
        Dose Response Model to plot
    dose_max:
        limit of the dose axis (in linear scale). Default: min(dose_data)
    dose_max:
        limit of the dose axis (in linear scale). Default: max(dose_data)
    model_colour:
        Colour of the model prediction.
        (See https://matplotlib.org/stable/gallery/color/named_colors.html)
    data_color:
        Colour of the dose-response data.
        (See https://matplotlib.org/stable/gallery/color/named_colors.html)
    title:
        title of the plot. Default is no title.
    x_axis_label:
        label of the x axis.
    y_axis_label:
        label of the y axis.
    dose_axis_scale:
        scale of the dose axis. Possible Values: 'lin', 'log'
    """

    if dose_response_model.parameters is None:
        raise ValueError("Dose Response Models need parameters for plotting.")

    # test if axis limits need to/can be derived
    if (dose_min is None or dose_max is None) and \
            dose_response_model.dose_data is None:
        raise ValueError("When plotting, a Dose Response Model must either "
                         "contain response data or `dose_min` and `dose_max` "
                         "must be set.")

    # set limits
    if dose_min is None and dose_axis_scale is 'lin':
        dose_min = 0
    else:
        dose_min = np.min(dose_response_model.dose_data)

    if dose_max is None:
        dose_max = np.max(dose_response_model.dose_data)

    # set x_axis
    if dose_axis_scale is 'lin':
        x_axis = np.linspace(start=dose_min,
                             stop=dose_max,
                             num=100)
    elif dose_axis_scale is 'log':
        # linear spacing in log scale
        x_axis = np.exp(
            np.linspace(start=np.log(dose_min),
                        stop=np.log(dose_max),
                        num=100))
    else:
        raise ValueError(f"Unknown axis scale {dose_axis_scale}. "
                         f"dose_axis_scale must be 'lin' or 'log'.")

    # plot the model prediction
    response_prediction = dose_response_model.get_multiple_responses(x_axis)
    model_plot = plt.plot(x_axis,
                          response_prediction,
                          color=model_colour)

    # plot the dose response data
    data_plot = None
    if (dose_response_model.dose_data is not None) and \
            (dose_response_model.response_data is not None):

        data_plot =plt.plot(dose_response_model.dose_data,
                            dose_response_model.response_data,
                            'x',
                            color=data_colour)

    # set axis scale
    if dose_axis_scale is 'log':
        plt.xscale('log')

    # format labels
    if title is not None:
        plt.title(title)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    # set right and upper axis visibility to False
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def plot_sample_scatter(dose_response_model: DoseResponseModelBase,
                        title: str = None):
    """
    Plots a scatter plot of the dose response model.
    Interfaces the corresponding pypesto visualization routine.
    """
    if dose_response_model.n_samples == 0:
        raise RuntimeError(
            'You must perform sampling before visualization. '
            'Run dose_response_model.sample_parameters(...)` first.')

    visualize.sampling_scatter(dose_response_model.pypesto_result,
                               suptitle=title)


def plot_1d_sample_marginals(dose_response_model: DoseResponseModelBase,
                             title: str = None):
    """
    Plots the 1d marginals of the dose response model.
    Interfaces the corresponding pypesto visualization routine.
    """
    if dose_response_model.n_samples == 0:
        raise RuntimeError(
            'You must perform sampling before visualization. '
            'Run `dose_response_model.sample_parameters(...)` first.')

    visualize.sampling_1d_marginals(dose_response_model.pypesto_result,
                                    suptitle=title)
