import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from SynRaptor import Combination
from SynRaptor import drug

number_of_iterations = 1000
null_model = 'bliss'
significances = np.zeros(number_of_iterations)

sigma2 = 0.0062

drug_a_parameters = np.array([4.88950696, 2.03546997, 0.48963928])
drug_b_parameters = np.array([9.58119493e-04, 1.88350336e+00, 3.27858417e-01])

drug_a_doses = np.array([4.5e-03, 4.5e-03, 4.5e-03, 4.5e-03, 4.5e-03, 4.5e-03, 1.4e-02,
                         1.4e-02, 1.4e-02, 1.4e-02, 1.4e-02, 1.4e-02, 4.0e-02, 4.0e-02,
                         4.0e-02, 4.0e-02, 4.0e-02, 4.0e-02, 1.2e-01, 1.2e-01, 1.2e-01,
                         1.2e-01, 1.2e-01, 1.2e-01, 3.7e-01, 3.7e-01, 3.7e-01, 3.7e-01,
                         3.7e-01, 3.7e-01, 1.1e+00, 1.1e+00, 1.1e+00, 1.1e+00, 1.1e+00,
                         1.1e+00, 3.4e+00, 3.4e+00, 3.4e+00, 3.4e+00, 3.4e+00, 3.4e+00,
                         1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 1.0e+01, 4.5e-03,
                         4.5e-03, 4.5e-03, 1.4e-02, 1.4e-02, 1.4e-02, 4.0e-02, 4.0e-02,
                         4.0e-02, 1.2e-01, 1.2e-01, 1.2e-01, 3.7e-01, 3.7e-01, 3.7e-01,
                         1.1e+00, 1.1e+00, 1.1e+00, 3.4e+00, 3.4e+00, 3.4e+00, 1.0e+01,
                         1.0e+01, 1.0e+01])

drug_b_doses = np.array([4.5e-06, 4.5e-06, 4.5e-06, 4.5e-06, 4.5e-06, 4.5e-06, 1.4e-05,
                         1.4e-05, 1.4e-05, 1.4e-05, 1.4e-05, 1.4e-05, 4.0e-05, 4.0e-05,
                         4.0e-05, 4.0e-05, 4.0e-05, 4.0e-05, 1.2e-04, 1.2e-04, 1.2e-04,
                         1.2e-04, 1.2e-04, 1.2e-04, 3.7e-04, 3.7e-04, 3.7e-04, 3.7e-04,
                         3.7e-04, 3.7e-04, 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03, 1.1e-03,
                         1.1e-03, 3.4e-03, 3.4e-03, 3.4e-03, 3.4e-03, 3.4e-03, 3.4e-03,
                         1.0e-02, 1.0e-02, 1.0e-02, 1.0e-02, 1.0e-02, 1.0e-02, 4.5e-06,
                         4.5e-06, 4.5e-06, 1.4e-05, 1.4e-05, 1.4e-05, 4.0e-05, 4.0e-05,
                         4.0e-05, 1.2e-04, 1.2e-04, 1.2e-04, 3.7e-04, 3.7e-04, 3.7e-04,
                         1.1e-03, 1.1e-03, 1.1e-03, 3.4e-03, 3.4e-03, 3.4e-03, 1.0e-02,
                         1.0e-02, 1.0e-02])

true_drug_a = drug.Drug(np.array([1]), np.array([1]), False, 1)
true_drug_b = drug.Drug(np.array([1]), np.array([1]), False, 1)
true_drug_a.parameters = drug_a_parameters
true_drug_b.parameters = drug_b_parameters

# Combination data that stays the same

comb_doses_a = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                         0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 1.08, 1.08,
                         1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08,
                         1.08, 1.08, 1.08, 1.08, 1.08, 3.25, 3.25, 3.25, 3.25,
                         3.25, 3.25, 3.25, 3.25, 3.25, 3.25, 3.25, 3.25, 3.25,
                         3.25, 3.25, 3.25, 10., 10., 10., 10., 10., 10.,
                         10., 10., 10., 10., 10., 10., 10., 10., 10.,
                         10.])

comb_doses_b = np.array([0.00011, 0.00011, 0.00011, 0.00011, 0.0005, 0.0005, 0.0005,
                         0.0005, 0.00223, 0.00223, 0.00223, 0.00223, 0.01, 0.01,
                         0.01, 0.01, 0.00011, 0.00011, 0.00011, 0.00011, 0.0005,
                         0.0005, 0.0005, 0.0005, 0.00223, 0.00223, 0.00223, 0.00223,
                         0.01, 0.01, 0.01, 0.01, 0.00011, 0.00011, 0.00011,
                         0.00011, 0.0005, 0.0005, 0.0005, 0.0005, 0.00223, 0.00223,
                         0.00223, 0.00223, 0.01, 0.01, 0.01, 0.01, 0.00011,
                         0.00011, 0.00011, 0.00011, 0.0005, 0.0005, 0.0005, 0.0005,
                         0.00223, 0.00223, 0.00223, 0.00223, 0.01, 0.01, 0.01,
                         0.01])
dose_combination = np.array([comb_doses_a, comb_doses_b])
for i in range(number_of_iterations):
    init_drug = drug.Drug(None, None, False, 1)

    drug_a_responses = init_drug.get_multiple_responses(drug_a_doses, drug_a_parameters, False) + \
                       np.random.normal(loc=0.0, scale=np.sqrt(sigma2), size=(1 * len(drug_a_doses)))
    drug_a = drug.Drug(drug_a_doses, drug_a_responses, False, 1)
    drug_a.fit_parameters()

    drug_b_responses = init_drug.get_multiple_responses(drug_b_doses, drug_b_parameters, False) + \
                       np.random.normal(loc=0.0, scale=np.sqrt(sigma2), size=(1 * len(drug_b_doses)))
    drug_b = drug.Drug(drug_b_doses, drug_b_responses, False, 1)
    drug_b.fit_parameters()

    true_combination = Combination([true_drug_a, true_drug_b])
    comb = Combination([drug_a, drug_b])
    get_combination_response = true_combination.combination_response(null_model)

    # create synthetic data

    synthetic_data = [get_combination_response(dose_combination[:, i], False, None) +
                      np.random.normal(loc=0.0, scale=np.sqrt(sigma2), size=None) for i in range(len(dose_combination))]

    # calculate significance

    significance = comb.volume_significance(dose_combination, synthetic_data, null_model)

    # create result array
    significances[i] = significance

plt.title('Bliss Volume Synthetic Data Test', fontsize=25)
plt.xlabel('significance levels', fontsize=20)
plt.ylabel('number of combinations', fontsize=20)

plt.hist(significances, bins=20)
plt.show()