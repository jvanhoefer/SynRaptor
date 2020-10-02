import matplotlib.pyplot as plt
import numpy as np
from SynRaptor import drug
from SynRaptor import plotting
import pandas as pd
import dictionaries
import math
from SynRaptor import Combination
from SynRaptor import drug_comb_dict as dct
import scipy as sp


def surface_plot(comb: Combination,
                 comb_doses_a,
                 comb_doses_b,
                 comb_responses,
                 null_model: str = 'bliss',
                 color: str = 'blue'):
    get_combination_response = comb.combination_response(null_model)

    fig = plt.figure()

    ax = fig.add_subplot(121, projection='3d')
    #title = null_model + ' prediction and responses'
    #ax.set_title(title, fontsize=17)

    x = np.arange(0, 10.3, 0.25)  # this is optimized for 5-FU
    y = np.arange(0, 0.012, 0.0005)  # this is optimized for MK-8669
    x, y = np.meshgrid(x, y)

    predictions = np.array([[get_combination_response(np.array([x[j][i], y[j][i]]), False, None) for i in
                             range(len(x[0]))] for j in range(len(x[:, 0]))])

    ax.scatter(comb_doses_a, comb_doses_b, comb_responses, color='k', marker='o')

    # here it is possible to plot the single drug data as well
    # ax.scatter(drug_a_doses, np.zeros(len(drug_a_doses)), drug_a_responses, color='green', marker='o')
    # ax.scatter(np.zeros(len(drug_b_doses)), drug_b_doses, drug_b_responses, color='r', marker='o')

    ax.set_xlabel('dose of 5-FU', fontsize=14)
    ax.set_ylabel('dose of MK-8669', fontsize=14)
    ax.set_zlabel('response', fontsize=14)

    ax.plot_surface(x, y, predictions, color=color, alpha=0.5)

    ax = fig.add_subplot(122)  # significances plot

    a_labels = ['0.35', '1.08', '3.25', '10.0']
    b_labels = ['0.00011', '0.0005', '0.00223', '0.01']

    significances = np.zeros(16)

    for i in range(16):
        a_dose = comb_doses_a[4 * i]
        b_dose = comb_doses_b[4 * i]
        responses = [comb_responses[4 * i + j] for j in range(4)]

        significances[i] = comb.get_significance(np.array([a_dose, b_dose]), responses, null_model)

    significances = np.reshape(significances, (4, 4))
    print('significances: ', significances)

    #print(significances)

    # fig, ax = plt.subplots()
    #im = ax.imshow(np.log(significances))
    im = ax.imshow(significances)# TODO choose between log or linear scale

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(a_labels)))
    ax.set_yticks(np.arange(len(b_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(a_labels, fontsize=13)
    ax.set_yticklabels(b_labels, fontsize=13)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('log of significance', rotation=-90, va="bottom", fontsize=17)
    #title = null_model + ' significance levels'
    #ax.set_title(title, fontsize=17)
    fig.tight_layout()
    plt.show()


def combination_plot(comb: Combination,
                 comb_doses_a,
                 comb_doses_b,
                 comb_responses,
                 null_model: str = 'bliss',
                 color: str = 'blue'):
    get_combination_response = comb.combination_response(null_model)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    #title = null_model + ' prediction and responses'
    #ax.set_title(title, fontsize=26)

    x = np.arange(0, 10.3, 0.25)  # this is optimized for 5-FU
    y = np.arange(0, 0.012, 0.0005)  # this is optimized for MK-8669
    x, y = np.meshgrid(x, y)

    predictions = np.array([[get_combination_response(np.array([x[j][i], y[j][i]]), False, None) for i in
                             range(len(x[0]))] for j in range(len(x[:, 0]))])

    ax.scatter(comb_doses_a, comb_doses_b, comb_responses, color='k', marker='o')

    # here it is possible to plot the single drug data as well
    # ax.scatter(drug_a_doses, np.zeros(len(drug_a_doses)), drug_a_responses, color='green', marker='o')
    # ax.scatter(np.zeros(len(drug_b_doses)), drug_b_doses, drug_b_responses, color='r', marker='o')

    ax.set_xlabel('dose 5-FU', fontsize=20)
    ax.set_ylabel('dose MK-8669', fontsize=20)
    ax.set_zlabel('response', fontsize=20)

    ax.plot_surface(x, y, predictions, color=color, alpha=0.5)
    plt.show()


def drug_plot(D: drug):#TODO monotone increasing
    """
    Plots the Hill curve according to parameters of Drug D and dose response data points of D.
    """
    fig = plt.figure('plot_drug')
    ax = fig.add_subplot(121)
    #parameters plot
    if D.parameters is not None:
        a = D.parameters[0]
        n = D.parameters[1]
        s = D.parameters[2]
        x = np.linspace(0, 2 * a, 100)  # 100 values from 0 to 2*a, as a is the Half-Max of Hill curve.
        y = D.control_response - s * x ** n / (a ** n + x ** n)
        plt.plot(x, y, label='parameters')

    #responses plot
    #D.dose_data.sort()
    #D.response_data.sort()#TODO numpy.argsort
    plt.plot(D.dose_data, D.response_data, linestyle='None', marker='.', label='responses')

    #design
    plt.title("linear scale")
    plt.xlabel('dose')
    plt.ylabel('response')
    plt.legend
    ax = fig.add_subplot(122)
    # parameters plot
    if D.parameters is not None:
        a = D.parameters[0]
        n = D.parameters[1]
        s = D.parameters[2]
        x = np.linspace(0, 2 * a, 100)  # 100 values from 0 to 2*a, as a is the Half-Max of Hill curve.
        y = D.control_response - s * x ** n / (a ** n + x ** n)
        plt.plot(x, y, label='parameters')

    # responses plot
    # D.dose_data.sort()
    # D.response_data.sort()#TODO numpy.argsort
    plt.semilogx(D.dose_data, D.response_data, linestyle='None', marker='.')

    # design
    plt.title("logarithmic scale")
    plt.xlabel('dose')
    plt.ylabel('response')
    plt.legend

    plt.show()