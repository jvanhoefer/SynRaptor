import matplotlib.pyplot as plt
import numpy as np
from SynRaptor import drug
from SynRaptor import plotting
from SynRaptor import figures
from SynRaptor import Combination
# from SynRaptor import drug_comb_dict as dct
import scipy as sp
# import pandas as pd
# import dictionaries
import math
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

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
comb = Combination([drug_a, drug_b])

comb_responses = np.array([0.93339, 0.94425, 1.04891, 0.87357, 0.85792, 0.8815, 0.82624,
                           0.79271, 0.66017, 0.63208, 0.66879, 0.65039, 0.61066, 0.56125,
                           0.57622, 0.49975, 0.7159, 0.95975, 0.87508, 0.95223, 0.66012,
                           0.82805, 0.70756, 0.76596, 0.2928, 0.69079, 0.58527, 0.53649,
                           0.56234, 0.6296, 0.52155, 0.51398, 0.71438, 0.70725, 0.741,
                           0.78635, 0.56747, 0.65913, 0.46289, 0.61324, 0.57185, 0.53188,
                           0.49668, 0.51762, 0.54672, 0.49466, 0.47421, 0.44041, 0.55738,
                           0.54691, 0.57181, 0.56162, 0.46394, 0.48216, 0.49246, 0.46075,
                           0.42699, 0.3056, 0.34517, 0.39684, 0.36953, 0.3684, 0.39696,
                           0.36154])

"""
fig = plt.figure(figsize=[6.4, 10])
ax = fig.add_subplot(121, projection='3d', azim=-60, elev=30)

x = np.arange(0, 10.3, 0.25)
y = np.arange(0, 0.012, 0.0005)
x, y = np.meshgrid(x, y)

bliss_predictions = np.array([[comb.get_bliss_response(np.array([x[j][i], y[j][i]]), False, None) for i in
                              range(len(x[0]))] for j in range(len(x[:, 0]))])
#hand_predictions = np.array([[comb.get_hand_response(np.array([x[j][i], y[j][i]]), False, None) for i in
#                              range(len(x[0]))] for j in range(len(x[:, 0]))])
#hsa_predictions = np.array([[comb.get_hsa_response(np.array([x[j][i], y[j][i]]), False, None) for i in
#                             range(len(x[0]))] for j in range(len(x[:, 0]))])
#loewe_predictions = np.array([[comb.get_loewe_response(np.array([x[j][i], y[j][i]]), False, None) for i in
#                               range(len(x[0]))] for j in range(len(x[:, 0]))])

ax.scatter(comb_doses_a, comb_doses_b, comb_responses, color='blue', marker='o')
#ax.scatter(drug_a_doses, np.zeros(len(drug_a_doses)), drug_a_responses, color='green', marker='o')
#ax.scatter(np.zeros(len(drug_b_doses)), drug_b_doses, drug_b_responses, color='r', marker='o')


ax.set_xlabel('5-FU (µM)')
ax.set_ylabel('MK-8669 (µM)')
ax.set_zlabel('viability')


ax.plot_surface(x, y, bliss_predictions, color='blue', alpha=0.3)
#ax.plot_surface(x, y, hand_predictions, color='green', alpha=0.3)
#ax.plot_surface(x, y, hsa_predictions, color='r', alpha=0.5)
#ax.plot_surface(x, y, loewe_predictions, color='yellow', alpha=0.8)

mean = np.zeros(16)
a_dose = np.zeros(16)
b_dose = np.zeros(16)


for i in range(16):
    a_dose[i] = comb_doses_a[4 * i]
    b_dose[i] = comb_doses_b[4 * i]
    responses = [comb_responses[4 * i + j] for j in range(4)]
    mean[i] = np.mean(responses)

a_dose = a_dose.reshape((4,4))
b_dose = b_dose.reshape((4,4))
mean = mean.reshape((4,4))


ax.plot_surface(a_dose, b_dose, mean, color='green', alpha=0.7)


#plt.show()

"""
"""
ax = fig.add_subplot(122)
# significances plot
# TODO write function in figures for all drugs and use it in figures_script
a_labels = ['0.35', '1.08', '3.25', '10.0']
b_labels = ['0.00011', '0.0005', '0.00223', '0.01']

significances = np.zeros(16)

for i in range(16):
    a_dose = [comb_doses_a[4 * i + j] for j in range(4)]
    b_dose = [comb_doses_b[4 * i + j] for j in range(4)]
    responses = [comb_responses[4 * i + j] for j in range(4)]
    significances[i] = comb.get_significance(np.array([a_dose, b_dose]), responses, 'bliss')
    # significances[i] = "{:.4f}".format(significance)

significances = np.reshape(significances, (4, 4))

#fig, ax = plt.subplots()
im = ax.imshow(np.log(significances))

# We want to show all ticks...
ax.set_xticks(np.arange(len(a_labels)))
ax.set_yticks(np.arange(len(b_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(a_labels)
ax.set_yticklabels(b_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# TODO text is to large
#for i in range(len(b_labels)):
#    for j in range(len(a_labels)):
#        text = ax.text(j, i, significances[i, j],
#                       ha="center", va="center", color="w")


cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('log of significance', rotation=-90, va="bottom")

ax.set_title("bliss significance levels")
fig.tight_layout()
plt.show()
print('ende')
"""
