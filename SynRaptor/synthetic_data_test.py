import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from SynRaptor import Combination
from SynRaptor import drug
from SynRaptor import figures

number_of_iterations = 100
null_model = 'bliss'
data_hsa = np.zeros(number_of_iterations)
significances = np.zeros(number_of_iterations)

sigma2 = 1 #TODO ändern auf drugs

drug_a_parameters = np.array([4.88950696, 2.03546997, 0.48963928])
drug_b_parameters = np.array([2.958119493, 4.88350336, 0.87858417])

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
for i in range(number_of_iterations):
    init_drug = drug.Drug(None, None, False, 1)
    drug_a_responses = init_drug.get_multiple_responses(drug_a_doses, drug_a_parameters, False) + \
        np.random.normal(loc=0.0, scale=np.sqrt(sigma2))
    drug_a = drug.Drug(drug_a_doses, drug_a_responses, False, 1)
    drug_a.fit_parameters()

    drug_b_responses = init_drug.get_multiple_responses(drug_b_doses, drug_b_parameters, False) + \
                       np.random.normal(loc=0.0, scale=np.sqrt(sigma2))
    drug_b = drug.Drug(drug_b_doses, drug_b_responses, False, 1)
    drug_b.fit_parameters()

    true_combination = Combination([true_drug_a, true_drug_b])
    comb = Combination([drug_a, drug_b])
    get_combination_response = true_combination.combination_response(null_model)

    # doses of drug a in combination ['0.35', '1.08', '3.25', '10.0']
    # doses of drug b in combination ['0.00011', '0.0005', '0.00223', '0.01']

    dose_combination = np.array([0.35, 0.00223])

    # create synthetic data, fit single drugs,

    synthetic_data = [get_combination_response(dose_combination, False, None) +
                      np.random.normal(loc=0.0, scale=np.sqrt(sigma2), size=None) for i in range(4)]

    # calculate significance

    significance = comb.get_significance(dose_combination, synthetic_data, null_model)
    """
    if significance < 0.8:
        if significance > 0.1:
            figures.drug_plot(drug_a)
            figures.drug_plot(drug_b)
            figures.combination_plot(comb, np.array([0.35, 0.35, 0.35, 0.35]),
                                 np.array([0.00223, 0.00223, 0.00223, 0.00223]), synthetic_data, 'bliss', 'blue')
    """

    # create result arrays

    significances[i] = significance

plt.hist(significances, bins=20)
plt.show()
