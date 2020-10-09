import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SynRaptor import Combination
from SynRaptor import drug_comb_dict as dct

significances = np.array([])
cell_line = 'A2058'
null_model = 'loewe'
drugs = dct.drug_dict()

# to use this script on another computer the file path needs to be adjusted below
combined_agents = pd.read_excel(
    "C:/Users/Carolin/PycharmProjects/GitHub/SynRaptor/SynRaptor/combined_agent_response.xls")

cl = combined_agents[combined_agents['cell_line'] == cell_line]
combination_names = cl['combination_name'].unique()

for i in range(len(combination_names)):
    new_combination = cl.loc[cl['combination_name'] == combination_names[i]]

    drug_a_name = new_combination.iloc[0, 2]
    drug_b_name = new_combination.iloc[0, 4]
    responses = np.array([])
    doses_a = np.array([])
    doses_b = np.array([])

    for index, row in new_combination.iterrows():
        for row_name in ['viability1', 'viability2', 'viability3', 'viability4']:
            if not math.isnan(row[row_name]):
                responses = np.append(responses, row[row_name])
                doses_a = np.append(doses_a, row['drugA Conc (µM)'])
                doses_b = np.append(doses_b, row['drugB Conc (µM)'])

    drug_a = drugs[(drug_a_name, cell_line)]
    drug_b = drugs[(drug_b_name, cell_line)]
    comb = Combination([drug_a, drug_b])

    dose_combinations = np.array([doses_a, doses_b])

    # calculate volume significances

    sig = comb.volume_significance(dose_combinations, responses, null_model)

    significances = np.append(significances, sig)


plt.title('Bliss Volume Significances for cell line A2058', fontsize=25)
plt.xlabel('significance levels', fontsize=20)
plt.ylabel('number of combinations', fontsize=20)
plt.hist(significances, bins=20)
plt.show()

