import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from SynRaptor import Combination
from SynRaptor import drug

number_of_iterations = 100
null_model = 'bliss'
data_hsa = np.zeros(number_of_iterations)
significances = np.zeros(number_of_iterations)

# Combination data that stays the same
for i in range(number_of_iterations):
    init_drug = drug.Drug()

    drug_a_parameters = np.array([4.88950696, 2.03546997, 0.48963928])
    drug_a_doses = 4.88950696 * 2 * np.random.random(10)
    drug_a_responses = init_drug.get_multiple_responses(drug_a_doses, drug_a_parameters, False)
    drug_a = drug.Drug(drug_a_doses, drug_a_responses, False, 1)
    drug_a.fit_parameters()

    drug_b_parameters = np.array([9.58119493e-04, 1.88350336e+00, 3.27858417e-01])
    drug_b_doses = 4.88950696 * 2 * np.random.random(10)
    drug_b_responses = init_drug.get_multiple_responses(drug_b_doses, drug_b_parameters, False)
    drug_b = drug.Drug(drug_b_doses, drug_b_responses, False, 1)
    drug_b.fit_parameters()

    comb = Combination([drug_a, drug_b])
    get_combination_response = comb.combination_response(null_model)

    # doses of drug a in combination ['0.35', '1.08', '3.25', '10.0']
    # doses of drug b in combination ['0.00011', '0.0005', '0.00223', '0.01']

    dose_combination = np.array([[0.35], [0.00223]])

    # create synthetic data, fit single drugs,

    synthetic_data = [get_combination_response(dose_combination, False, None) +
                     np.random.normal(loc=0.0, scale=np.sqrt(comb.sigma2), size=None) for i in range(4)]

    # calculate significance

    significance = comb.get_significance(dose_combination, synthetic_data, null_model)

    # create result arrays

    significances[i] = significance

plt.hist(significances, bins=20)
plt.show()
