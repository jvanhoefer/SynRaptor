import matplotlib.pyplot as plt
import numpy as np
from SynRaptor import drug
from SynRaptor import plotting
import pandas as pd
import dictionaries

single_agents = pd.read_excel("single_agent_response.xlsx")
drug_names = single_agents['drug_name'].unique()
drug_list = np.array([])
drug_dict = dict()

for i in range(len(drug_names)):
    new_drug = single_agents.loc[single_agents['drug_name'] == drug_names[i]][['drug_name', 'Drug_concentration (µM)', \
                                                                            'viability1', 'viability2','viability3', \
                                                                            'viability4', 'viability5', 'viability6']]

    responses = np.array([])
    doses = np.array([])

    for index, row in new_drug.iterrows():
        responses = np.append(responses, row['viability1'])
        doses = np.append(doses,row['Drug_concentration (µM)'])
        if not row['viability2'] == '':
            responses = np.append(responses, row['viability2'])
            doses = np.append(doses, row['Drug_concentration (µM)'])
        if not row['viability3'] == '':
            responses = np.append(responses, row['viability3'])
            doses = np.append(doses, row['Drug_concentration (µM)'])
        if not row['viability4'] == '':
            responses = np.append(responses, row['viability4'])
            doses = np.append(doses, row['Drug_concentration (µM)'])
        if not row['viability5'] == '':
            responses = np.append(responses, row['viability5'])
            doses = np.append(doses, row['Drug_concentration (µM)'])
        if not row['viability6'] == '':
            responses = np.append(responses, row['viability6'])
            doses = np.append(doses, row['Drug_concentration (µM)'])

        drug.Drug(doses, responses)

        drug_dict.update({drug_names[i] : drug.Drug(doses, responses)})




