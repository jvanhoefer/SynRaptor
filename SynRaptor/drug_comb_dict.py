import math
import numpy as np
import pandas as pd
from SynRaptor import drug


# to use this on another computer the file paths need to be adjusted below

def drug_dict():
    single_agents = pd.read_excel(
        "C:/Users/Carolin/PycharmProjects/GitHub/SynRaptor/SynRaptor/single_agent_response.xlsx")
    drug_names = single_agents['drug_name'].unique()
    drug_dict = dict()

    for i in range(len(drug_names)):
        cl = single_agents[single_agents['drug_name'] == drug_names[i]]
        cell_lines = cl['cell_line'].unique()

        for j in range(len(cell_lines)):
            new_drug = cl.loc[cl['cell_line'] == cell_lines[j]]
            responses = np.array([])
            doses = np.array([])

            for index, row in new_drug.iterrows():
                for row_name in ['viability1', 'viability2', 'viability3', 'viability4', 'viability5', 'viability6']:
                    if not math.isnan(row[row_name]):
                        responses = np.append(responses, row[row_name])
                        doses = np.append(doses, row['Drug_concentration (µM)'])

            d = drug.Drug(doses, responses, False, 1)
            d.fit_parameters()
            drug_dict.update({(drug_names[i], cell_lines[j]): d})
    return drug_dict


def comb_dict():
    combined_agents = pd.read_excel(
        "C:/Users/Carolin/PycharmProjects/GitHub/SynRaptor/SynRaptor/combined_agent_response.xls")
    combination_names = combined_agents['combination_name'].unique()
    comb_dict = dict()

    for i in range(len(combination_names)):
        cl = combined_agents[combined_agents['combination_name'] == combination_names[i]]
        cell_lines = cl['cell_line'].unique()
        for j in range(len(cell_lines)):
            new_combination = cl.loc[cl['cell_line'] == cell_lines[j]][['drugA Conc (µM)', 'drugB Conc (µM)', \
                                                                        'viability1', 'viability2', 'viability3', \
                                                                        'viability4']]

            responses = np.array([])
            doses_a = np.array([])
            doses_b = np.array([])

            for index, row in new_combination.iterrows():
                for row_name in ['viability1', 'viability2', 'viability3', 'viability4']:
                    if not math.isnan(row[row_name]):
                        responses = np.append(responses, row[row_name])
                        doses_a = np.append(doses_a, row['drugA Conc (µM)'])
                        doses_b = np.append(doses_b, row['drugB Conc (µM)'])

                comb_dict.update({(combination_names[i], cell_lines[j], 'responses'): responses})
                comb_dict.update({(combination_names[i], cell_lines[j], 'doses_a'): doses_a})
                comb_dict.update({(combination_names[i], cell_lines[j], 'doses_b'): doses_b})
    return comb_dict


def get_combination_data(combination_name: str,
                         cell_line):
    combined_agents = pd.read_excel(
        "C:/Users/Carolin/PycharmProjects/GitHub/SynRaptor/SynRaptor/combined_agent_response.xls")

    cl = combined_agents[combined_agents['combination_name'] == combination_name]

    new_combination = cl.loc[cl['cell_line'] == cell_line][['drugA Conc (µM)', 'drugB Conc (µM)', \
                                                            'viability1', 'viability2', 'viability3', \
                                                            'viability4']]

    responses = np.array([])
    doses_a = np.array([])
    doses_b = np.array([])

    for index, row in new_combination.iterrows():
        for row_name in ['viability1', 'viability2', 'viability3', 'viability4']:
            if not math.isnan(row[row_name]):
                responses = np.append(responses, row[row_name])
                doses_a = np.append(doses_a, row['drugA Conc (µM)'])
                doses_b = np.append(doses_b, row['drugB Conc (µM)'])

    return doses_a, doses_b, responses


def create_drug(drug_name: str,
                cell_line):
    single_agents = pd.read_excel(
        "C:/Users/Carolin/PycharmProjects/GitHub/SynRaptor/SynRaptor/single_agent_response.xlsx")

    cl = single_agents[single_agents['drug_name'] == drug_name]
    new_drug = cl.loc[cl['cell_line'] == cell_line]
    responses = np.array([])
    doses = np.array([])

    for index, row in new_drug.iterrows():
        for row_name in ['viability1', 'viability2', 'viability3', 'viability4', 'viability5', 'viability6']:
            if not math.isnan(row[row_name]):
                responses = np.append(responses, row[row_name])
                doses = np.append(doses, row['Drug_concentration (µM)'])

    created_drug = drug.Drug(doses, responses, False, 1)
    created_drug.fit_parameters()

    return created_drug
