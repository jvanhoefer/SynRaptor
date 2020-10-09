import matplotlib.pyplot as plt
import numpy as np
from SynRaptor import drug

"""
plot_responses(drug: Drug):
    Plots the dose and response data of a drug drug.
    
plot_parameters(drug: Drug):
    Plots a Hill curve according to the parameters of a Drug drug.
    
plot_noise(drug: Drug):
    Visualises the accuracy of fit_parameters().
    
noise_response(D: Drug):
    Generates and stores response_data to dose_data through adding gaussian noise on results of get_multiple_responses.
    
plot_drug(D: Drug):
    Plots the Hill curve according to parameters of Drug D and dose response data points of D.

"""


def plot_responses(drug: drug):
    """
    Plots the dose and response data of a drug drug.
    """
    plt.figure('plot_responses')
    plt.semilogx(drug.dose_data, drug.response_data, linestyle='None', marker='.')

    # design
    plt.title('dose response curve')
    plt.xlabel('dose')
    plt.ylabel('response')
    plt.show()


def plot_parameters(drug: drug):  # TODO monotone increasing
    """
    Plots a Hill curve according to the parameters of a Drug drug.
    If drug does not have parameters yet, the parameters will be determined via drug.fit_parameters() and stored in drug
    """
    plt.figure('plot_parameters')
    if drug.parameters is None:
        D.fit_parameters()
    a = drug.parameters[0]
    n = drug.parameters[1]
    s = drug.parameters[2]
    x = np.linspace(0, 2 * a, 100)  # 100 values from 0 to 2*a, as a is the Half-Max of Hill curve.
    y = drug.control_response - s * x ** n / (a ** n + x ** n)
    plt.plot(x, y)

    # design
    plt.xlabel('dose')
    plt.ylabel('response')
    plt.title('Hill curve')
    plt.show()


def plot_noise(drug: drug):
    """
    Visualises the accuracy of fit_parameters().

    Three steps are plotted:
        1. The Hill curve according to the parameters of Drug drug.
        2. dose_response data points generated from the above Hill curve + 0.1 * gaussian noise. (using noise_response)
        3. The Hill curve of fitted parameters for in step 2 generated data points.

    """
    plt.figure('plot_noise')
    if drug.parameters is None:
        drug.fit_parameters(10)

    # parameters_plot
    a = drug.parameters[0]
    n = drug.parameters[1]
    s = drug.parameters[2]
    x = np.linspace(0, 2 * a, 100)  # 100 values from 0 to 2*a, as a is the Half-Max of Hill curve.
    y = drug.control_response + s * x ** n / (a ** n + x ** n)
    plt.plot(x, y, label='true parameters')

    # responses_plot
    noise_response(drug)
    plt.plot(drug.dose_data, drug.response_data, linestyle='None', marker='.', label='data')

    # new parameters plot
    D.fit_parameters(10)
    a = drug.parameters[0]
    n = drug.parameters[1]
    s = drug.parameters[2]
    x = np.linspace(0, 2 * a, 100)  # 100 values from 0 to 2*a, as a is the Half-Max of Hill curve.
    y = drug.control_response + s * x ** n / (a ** n + x ** n)
    plt.plot(x, y, linestyle='--', label='fitted hill curve')

    # design
    plt.title('Hill curves')
    plt.xlabel('dose')
    plt.ylabel('response')
    plt.legend()
    plt.show()


def noise_response(D: drug):
    """
    Generates and stores response_data to dose_data through adding gaussian noise on results of get_multiple_responses.
    """
    D.dose_data.sort()
    y = D.get_multiple_responses(D.dose_data, D.parameters)
    for i in range(len(D.dose_data)):
        y[i] = y[i] + 0.1 * np.random.normal()
    D.response_data = y


def plot_drug(D: drug, title: str = 'dose response curve'):  # TODO monotone increasing
    """
    Plots the Hill curve according to parameters of Drug D and dose response data points of D.
    """
    plt.figure('plot_drug')
    # parameters plot
    if D.parameters is not None:
        a = D.parameters[0]
        n = D.parameters[1]
        s = D.parameters[2]
        x = np.linspace(0, 2 * a, 100)  # 100 values from 0 to 2*a, as a is the Half-Max of Hill curve.
        y = D.control_response - s * x ** n / (a ** n + x ** n)
        plt.plot(x, y, label='parameters')

    # responses plot
    plt.plot(D.dose_data, D.response_data, linestyle='None', marker='.', label='responses')

    # design
    plt.title(title)
    plt.xlabel('dose')
    plt.ylabel('response')
    plt.legend
    plt.show()

