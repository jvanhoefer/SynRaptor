import matplotlib.pyplot as plt
import numpy as np
from SynRaptor import drug
from SynRaptor import plotting
import pandas as pd
import dictionaries
import math
from SynRaptor import Combination

single_agents = pd.read_excel("single_agent_response.xlsx")
drug_names = single_agents['drug_name'].unique()
drug_list = np.array([])
drug_dict = dict()

for i in range(len(drug_names)):
    cl = single_agents[single_agents['drug_name'] == drug_names[i]]
    cell_lines = cl['cell_line'].unique()
    for j in range(len(cell_lines)):
        new_drug = cl.loc[cl['cell_line'] == cell_lines[j]][['Drug_concentration (µM)', \
                                                             'viability1', 'viability2', 'viability3', \
                                                             'viability4', 'viability5', 'viability6']]

        responses = np.array([])
        doses = np.array([])

        for index, row in new_drug.iterrows():
            for row_name in ['viability1', 'viability2', 'viability3', 'viability4', 'viability5', 'viability6']:
                if not math.isnan(row[row_name]):
                    responses = np.append(responses, row[row_name])
                    doses = np.append(doses, row['Drug_concentration (µM)'])

            drug_dict.update({(drug_names[i], cell_lines[j]): drug.Drug(doses, responses, True, 0)})
            # TODO not all drugs are monotone increasing with w_0=0 as assumed here ^^^


# print('dict:',drug_dict)


def get_figures(drug_a_name: str,
                drug_b_name: str,
                cell_line: str):
    combi = pd.read_excel("combined_agent_response.xls")
    validation_data = combi.loc[(combi['cell_line'] == cell_line) & (combi['drugA_name'] == drug_a_name) &
                                (combi['drugB_name'] == drug_b_name)][
        ['drugA Conc (µM)', 'drugB Conc (µM)', 'viability1']]

    drug_a = drug_dict[(drug_a_name, cell_line)]
    drug_b = drug_dict[(drug_b_name, cell_line)]
    combination = Combination([drug_a, drug_b])
    alpha = combination.get_significance(np.array([validation_data.iloc[0, 0], validation_data.iloc[0, 1]]),
                                         validation_data.iloc[0, 2], 'hsa')
    return alpha


print(get_figures('5-FU', 'ABT-888', 'A2058'))
