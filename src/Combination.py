"""Combinations of multiple drugs are defined here.
"""
import numpy as np
import scipy as sp
from src import Drug


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
        if not self._check_drug_consistency:
            raise RuntimeError

        #self._set_sigma() TODO

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

            Checks requirements via check_bliss_requirements. For monotone increasing drugs
            the response is calculated as 1 - prod(1- single_response_i). For monotone decreasing drugs as
            prod(single_response_i). TODO wollen auch gradienten bzgl der einzelnen drug parameter haben.

            Parameters
            ----------
            dose_combination: np.array
                gives the doses for the drugs that will be combined

            gradient: bool
                determines wether gradient should be returned as well

            Returns
            -------
            response: float
                the response of the combined drug

            grad: np.array
                TODO not implemented yet
            """
            if not self._check_bliss_requirements(dose_combination):
                raise RuntimeError
            if drug_list[0].monotone_increasing:
                prod = 1
                for i in range(len(drug_list)):
                    prod *= 1 - drug_list[i].get_response(dose_combination[i])
                return 1 - prod
            else:
                prod = 1
                for i in range(len(drug_list)):
                    prod *= drug_list[i].get_response(dose_combination[i])
                return prod

    def _check_bliss_requirements(self,
                             dose_combination: np.array):
        """
        Checks if requirements for get_bliss_response are fulfilled.

        Checks if for monotone increasing drugs the control_response is 0, and the maximal effect is <= 1. As the
        control response is the same for all drugs it it sufficient to check for drug_list[0].
        For decreasing drugs the control_response has to be 1 and the parameter s of Hill curve <=1. If one requirement
        is not met, False is returned. Otherwise True is returned.
        """
        if drug_list[0].monotone_increasing:
            for i in range(len(drug_list)):
                if drug_list[i].parameters[2] > 1:
                    return False
            if not (drug_list[0].control_response == 0):
                return False
        else:
            for i in range(len(drug_list)):
                if drug_list[i].parameters[2] > 1:
                    return False
            if not (drug_list[0].control_response == 1):
                return False
        return True

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

            This function calculates the HSA response, taking into consideration wether the single dose effect curves are
            monotone increasing or not. For monotone increasing drugs the maximal response is returned. For monotone
            decreasing drugs the minimal response of the single drugs is returned. TODO gradient

            Parameters
            ----------
            dose_combination: np.array
                gives the doses for the drugs that will be combined

            gradient: bool
                determines wether gradient should be returned as well

            Returns
            -------
            response: float
                the response of the combined drug

            grad: np.array
                TODO not implemented yet
            """
            if drug_list[0].monotone_increasing:
                response = - float('inf')
                for i in range(len(drug_list)):
                    temp = drug_list[i].get_response(dose_combination[i])
                    if temp > response:
                        response = temp
                return response
            else:
                response = float('inf')
                for i in range(len(drug_list)):
                    temp = drug_list[i].get_response(dose_combination[i])
                    if temp < response:
                        response = temp
                return response


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
        If yes return True, if not return False.
        """
        l = len(drug_list)
        control_response = drug_list[l-1].control_response
        mon = drug_list[l-1].monotone_increasing
        for i in range(l-1):
            if not (drug_list[i].control_response == control_response):
                return False
            if not (drug_list[i].monotone_increasing == mon):
                return False
        return True


    def _set_sigma(self):
        """
        Compute the (optimal) sigma for the given drugs in drug_list
        #formel von hiererchischer optimierung für alle drugs zusammen
        """

        self.sigma = - float('nan')

        raise NotImplementedError

"""
This code may be used for testing.

x = np.array([1,2,3,4,5])
y = np.array([2,4,6,6,7])
z = np.array([1,2,4,5,7])


A = Drug.Drug(x,0.0001*y)
A.fit_parameters(10)


B = Drug.Drug(y,0.0001*z)
B.fit_parameters(10)


C = Drug.Drug(x,0.0001*z)
C.fit_parameters(10)


A.parameters[2] = 0.4
B.parameters[2] = 0.3
C.parameters[2] = 0.7

drug_list = [A,B,C]
Comb = Combination(drug_list)
dose = np.array([0.5,0.5,0.5])
res = Comb.get_hsa_response(dose, False)

res2 = Comb.get_bliss_response(dose, False)
print(res2)
"""