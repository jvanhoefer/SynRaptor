import matplotlib.pyplot as plt
import numpy as np
from SynRaptor import drug
from SynRaptor import plotting
from SynRaptor import figures
from SynRaptor import Combination
from SynRaptor import drug_comb_dict as dct
import scipy as sp
import pandas as pd
import dictionaries
import math

#insert drug names and cell line here:

drug_a_name = '5-FU'
drug_b_name = 'MK-8669'
combination_name = '5-FU & MK-8669'
cell_line = 'A2058'


#here the drug and combination objects are created

drug_a = dct.create_drug(drug_a_name, cell_line)
drug_b = dct.create_drug(drug_b_name, cell_line)
comb = Combination([drug_a, drug_b])
(comb_doses_a, comb_doses_b, comb_responses) = dct.get_combination_data(combination_name, cell_line)
dose_combination = np.array([comb_doses_a, comb_doses_b])


#here the single drugs are plotted

#plotting.plot_drug(drug_a, drug_a_name)
#plotting.plot_drug(drug_b, drug_b_name)


#here the significance level is printed

#print('Significance level per null_model:')
#print('Hand:', comb.get_significance(dose_combination, comb_responses, 'hand'))
#print('Bliss:', comb.get_significance(dose_combination, comb_responses, 'bliss'))
#print('HSA:', comb.get_significance(dose_combination, comb_responses, 'hsa'))
#print('Loewe:', comb.get_significance(dose_combination, comb_responses, 'loewe'))


#here isoboles are plotted

#figures.plot_validation_isoboles(0.95, 'hand', 0.05, comb)
#figures.plot_validation_isoboles(0.95, 'bliss', 0.05, comb)
#figures.plot_validation_isoboles(0.95, 'hsa', 0.05, comb)
"""
for bisection to work one has to be careful with choosing values for the effect and for alpha"""
#figures.plot_validation_isoboles(0.95, 'loewe', 0.05, comb)


#here the null model predictions are plotted

figures.surface_plot(comb, comb_doses_a, comb_doses_b, comb_responses, 'bliss')
figures.surface_plot(comb, comb_doses_a, comb_doses_b, comb_responses, 'hsa')
figures.surface_plot(comb, comb_doses_a, comb_doses_b, comb_responses, 'loewe')
figures.surface_plot(comb, comb_doses_a, comb_doses_b, comb_responses, 'hand')
