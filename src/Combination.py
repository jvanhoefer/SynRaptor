"""Combinations of multiple drugs are defined here.
"""
import numpy as np
import scipy as sp
from . import Drug


class Combination:
    """
    Combination stores a list of drugs and gives all functionality
    to evaluate synergy null models and derive synergy.

    """

    def __init__(self,
                 drug_list: list):
        """
        Constructor
        """

        # check if drugs are consistent... else raise Error...
        self.drug_list = drug_list

        self._set_sigma()

    def get_loewe_response(self,
                           dose_combination: np.array,
                           gradient: bool):
        """
        Compute the Loewe response
        """
        raise NotImplementedError

    def get_bliss_response(self,
                           dose_combination: np.array,
                           gradient: bool):
            """
            Compute the Bliss response
            """
            raise NotImplementedError

    def get_hand_response(self,
                          dose_combination: np.array,
                          gradient: bool):
            """
            Compute the hand response
            """
            raise NotImplementedError

    def get_hsa_response(self,
                         dose_combination: np.array,
                         gradient: bool):
            """
            Compute the HSA response
            """
            raise NotImplementedError

    def get_loewe_significance(self,
                               dose_combination: np.array,
                               responses: np.array):
        """
        Compute the Loewe significance level for a given dose and measurements.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        """
        raise NotImplementedError

    def get_bliss_significance(self,
                               dose_combination: np.array,
                               responses: np.array):
        """
        Compute the Bliss significance level for a given dose and measurements.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        """
        raise NotImplementedError

    def get_hand_significance(self,
                              dose_combination: np.array,
                              responses: np.array):
        """
        Compute the Hand significance level for a given dose and measurements.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        """
        raise NotImplementedError

    def get_hsa_significance(self,
                             dose_combination: np.array,
                             responses: np.array):
        """
        Compute the HSA significance level for a given dose and measurements.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        """
        raise NotImplementedError

    def _check_drug_consistency(self,
                                drug_list: list)->bool:
        """
        check, if all drugs are either mon incr or decr. ...
        check, if all drugs share the same w_0 ...
        """

        raise NotImplementedError

    def _set_sigma(self):
        """
        Compute the (optimal) sigma for the given drugs in drug_list
        """

        self.sigma = - float('nan')

        raise NotImplementedError
