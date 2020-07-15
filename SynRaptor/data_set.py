import matplotlib.pyplot as plt
import numpy as np
from SynRaptor import drug
from SynRaptor import plotting
import pandas as pd
import dictionaries
import math

single_agents = pd.read_excel("single_agent_response.xlsx")
drug_names = single_agents['drug_name'].unique()
drug_list = np.array([])
drug_dict = dict()

for i in range(len(drug_names)):
    cl = single_agents[single_agents['drug_name'] == drug_names[i]]
    cell_lines = cl['cell_line'].unique()
    for j in range(len(cell_lines)):
        new_drug = cl.loc[cl['cell_line'] == cell_lines[j]][['Drug_concentration (µM)', \
                                                            'viability1', 'viability2','viability3', \
                                                            'viability4', 'viability5', 'viability6']]

        responses = np.array([])
        doses = np.array([])

        for index, row in new_drug.iterrows():
            for row_name in ['viability1', 'viability2', 'viability3', 'viability4','viability5','viability6']:
                if not math.isnan(row[row_name]):
                    responses = np.append(responses, row[row_name])
                    doses = np.append(doses, row['Drug_concentration (µM)'])



            drug_dict.update({(drug_names[i], cell_lines[j]): drug.Drug(doses, responses, True, 0)})
            #TODO not all drugs are monotone increasing with w_0=0 as assumed here ^^^

print('dict:',drug_dict)






