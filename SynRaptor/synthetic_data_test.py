import matplotlib.pyplot as plt
import numpy as np
from SynRaptor import drug
from SynRaptor import plotting
from SynRaptor import figures
from SynRaptor import Combination
# from SynRaptor import drug_comb_dict as dct
import scipy as sp
import pandas as pd
# import dictionaries
import math
from scipy.stats import chi2

# Combination data that stays the same

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
drug_a_responses = np.array([0.90804, 1.00986, 1.01038, 1.18202, 0.95803, 1.03229, 0.92227,
                             1.08071, 0.97098, 1.06446, 1.03962, 1.16256, 0.97264, 1.11151,
                             1.05546, 1.10022, 1.03573, 1.06873, 0.92836, 1.07104, 0.95915,
                             1.10822, 1.04019, 1.14458, 1.02036, 1.11621, 1.02111, 1.0539,
                             0.95623, 1.02383, 0.96456, 0.98167, 0.95253, 0.98887, 0.88472,
                             1.0134, 0.71601, 0.86389, 0.7657, 0.75393, 0.79795, 0.81468,
                             0.55264, 0.6407, 0.56285, 0.53192, 0.5363, 0.5928, 1.05309,
                             1.15159, 1.00919, 0.99496, 1.06874, 0.99616, 1.00799, 0.92548,
                             0.94833, 1.00454, 1.0254, 0.92165, 1.03322, 1.03846, 0.99449,
                             0.91147, 0.97289, 1.04985, 1.0435, 0.95032, 0.87949, 0.65022,
                             0.6593, 0.69825])
drug_a_parameters = np.array([4.88950696, 2.03546997, 0.48963928])
drug_a = drug.Drug(drug_a_doses, drug_a_responses, False, 1)
drug_a.parameters = drug_a_parameters

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
drug_b_responses = np.array([0.99327, 0.97818, 0.95645, 1.16545, 0.97211, 1.16065, 0.96231,
                             1.083, 1.00223, 1.17331, 0.96641, 1.13004, 0.95903, 1.04938,
                             0.94805, 1.18283, 1.04446, 1.08758, 0.95394, 1.07815, 1.05277,
                             1.09432, 1.03596, 1.1003, 0.95667, 1.04434, 0.95738, 1.17461,
                             0.79636, 0.90663, 0.84165, 0.92711, 0.83811, 0.96631, 0.74605,
                             0.84718, 0.70854, 0.80214, 0.80615, 0.78408, 0.5947, 0.69404,
                             0.71504, 0.69508, 0.70939, 0.81031, 0.64612, 0.77982, 0.97113,
                             1.07171, 1.02647, 0.94589, 1.02624, 0.93822, 0.98748, 1.01536,
                             1.01992, 0.89716, 0.97536, 0.92134, 0.91753, 0.86726, 0.87879,
                             0.7738, 0.75992, 0.65828, 0.60355, 0.65835, 0.64123, 0.58291,
                             0.55332, 0.59048])
drug_b_parameters = np.array([9.58119493e-04, 1.88350336e+00, 3.27858417e-01])
drug_b = drug.Drug(drug_b_doses, drug_b_responses, False, 1)
drug_b.parameters = drug_b_parameters

comb = Combination([drug_a, drug_b])
(rss_y_y, theta_y) = comb.fit_to_old_data()

# doses of drug a in combination ['0.35', '1.08', '3.25', '10.0']
# doses of drug b in combination ['0.00011', '0.0005', '0.00223', '0.01']

dose_combination = np.array([[1.08], [0.00223]])
in_validation = 0
out_validation = 0
less1 = 0
less2 = 0
less3 = 0
less4 = 0
less5 = 0
more1 = 0
more2 = 0
more3 = 0
more4 = 0
more5 = 0

for i in range(100):
    # create synthetic data

    synthetic_data = comb.get_bliss_response(dose_combination, False, None) + \
                     np.random.normal(loc=0.0, scale=np.sqrt(comb.sigma2), size=None)


    # calculate significance

    rss_y_yz = rss_y_y + comb.evaluate_validation_residual(np.array([synthetic_data]), dose_combination, False,
                                                           np.reshape(theta_y, (-1, 3), order='C'), 'bliss')
    rss_yz_yz = comb.fit_to_full_data(np.array([synthetic_data]), dose_combination, 'bliss')
    difference = rss_y_yz - rss_yz_yz
    significance = chi2.cdf(difference, 1, loc=0, scale=np.sqrt(comb.sigma2))

    # analyze data

    #print(i, '. data:', synthetic_data, ' significance:', significance)





    if significance < 0.90:
        less1 += 1
    else:
        more1 += 1


    if significance < 0.80:
        less2 += 1
    else:
        more2 += 1

    if significance < 0.50:
        less3 += 1
    else:
        more3 += 1

    if significance < 0.30:
        less4 += 1
    else:
        more4 += 1

    if significance < 0.10:
        less5 += 1
    else:
        more5 += 1

#print('In validation interval: ', in_validation, ' Out of validation interval: ', out_validation)
# significance < 0.05
# printed for bliss: In validation interval:  444  Out of validation interval:  556
# for hsa: In validation interval:  499  Out of validation interval:  501
# In validation interval:  130  Out of validation interval:  870
# In validation interval:  132  Out of validation interval:  868
# In validation interval:  108  Out of validation interval:  892

# significance < 0.95
# In validation interval:  1000  Out of validation interval:  0

print('less than 90: ', less1, 'more than 90: ', more1)
print('less than 80: ', less2, 'more than 80: ', more2)
print('less than 50: ', less3, 'more than 50: ', more3)
print('less than 30: ', less4, 'more than 30: ', more4)
print('less than 10: ', less5, 'more than 10: ', more5)

# less than 90:  10000 more than 90:  0
# less than 80:  9988 more than 80:  12
# less than 50:  9510 more than 50:  490
# less than 30:  6977 more than 30:  3023
# less than 10:  2782 more than 10:  7218
