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


# TODO funktion die parameters fitted falls None


def fill_parameters(drug: drug):
    if drug.parameters is None:
        drug.fit_parameters(10)
    return


def get_figures(drug_a_name: str,
                drug_b_name: str,
                cell_line: str):
    combi = pd.read_excel("C:/Users/Carolin/PycharmProjects/GitHub/SynRaptor/SynRaptor/combined_agent_response.xls")
    validation_data = combi.loc[(combi['cell_line'] == cell_line) & (combi['drugA_name'] == drug_a_name) &
                                (combi['drugB_name'] == drug_b_name)][
        ['drugA Conc (µM)', 'drugB Conc (µM)', 'viability1']]  # TODO gibts die selbe kombi auch andersrum?

    drug_a = drug_dict[(drug_a_name, cell_line)]
    drug_b = drug_dict[(drug_b_name, cell_line)]
    combination = Combination([drug_a, drug_b])
    alpha = combination.get_significance(np.array([validation_data.iloc[0, 0], validation_data.iloc[0, 1]]),
                                         validation_data.iloc[0, 2], 'hsa')
    # TODO plot single drug

    return alpha


#print(get_figures('5-FU', 'MK-8669', 'A2058'))


def get_isobole(effect,
                null_model: str,
                comb: Combination):
    """
    Calculates isobole for given model and effect.

    """


    if null_model == 'bliss':
        get_combination_response = comb.get_bliss_response
    elif null_model == 'hsa':
        get_combination_response = comb.get_hsa_response
    elif null_model == 'loewe':
        get_combination_response = comb.get_loewe_response
    elif null_model == 'hand':
        get_combination_response = comb.get_hand_response
    else:
        ValueError("invalid null model")

    doses_a = np.array([])
    doses_b = np.array([])

    doses_a = np.append(doses_a, comb.drug_list[0].inverse_evaluate(effect))
    doses_b = np.append(doses_b, 0)

    ratio = [1 / 9, 1 / 4, 3 / 7, 4 / 6, 1, 6 / 4, 7 / 3, 4, 9]

    for r in ratio:
        def isobole_equation(c):
            dose_combination = np.array([c, r * c])
            return get_combination_response(dose_combination, False, None) - effect

        dose_a = sp.optimize.bisect(isobole_equation, 0, 10)

        doses_a = np.append(doses_a, dose_a)
        doses_b = np.append(doses_b, r * dose_a)

    doses_a = np.append(doses_a, 0)
    doses_b = np.append(doses_b, comb.drug_list[1].inverse_evaluate(effect))

    return doses_a, doses_b




def get_validation_interval(effect,
                            null_model: str,
                            alpha: float,
                            comb: Combination):
    """
    Calculates bound of validation interval for 10 dose ratios and returns arrays for plotting.
    """
    doses_a = np.array([])
    doses_b = np.array([])

    ratio = [0, 1 / 9, 1 / 4, 3 / 7, 4 / 6, 1, 6 / 4, 7 / 3, 4, 9]

    for r in ratio:
        def significance_equation(c):
            dose_combination = np.array([c, r * c])
            return comb.get_significance(np.array(dose_combination), effect, null_model) - alpha

        dose_a = sp.optimize.bisect(significance_equation, 0, 10)

        doses_a = np.append(doses_a, dose_a)
        doses_b = np.append(doses_b, r * dose_a)

    def significance_equation(c):
        dose_combination = np.array([0, c])
        return comb.get_significance(dose_combination, effect, null_model) - alpha

    dose_b = sp.optimize.bisect(significance_equation, 0, 10)

    doses_a = np.append(doses_a, 0)
    doses_b = np.append(doses_b, dose_b)

    return doses_a, doses_b


def plot_validation_isoboles(effect: float,
                             null_model: str,
                             alpha: float,
                             comb: Combination):
    """
    Creates image using get_isobole and get_validation_interval.
    """
    (isobole_x, isobole_y) = get_isobole(effect, null_model, comb)
    plt.plot(isobole_x, isobole_y, linestyle='None', marker='o', color='green', label='isobole')
    (validation_x, validation_y) = get_validation_interval(effect, null_model, alpha, comb)
    plt.plot(validation_x, validation_y, linestyle='None', marker='.', color='blue', label='validation')
    plt.title(null_model)
    plt.show()


