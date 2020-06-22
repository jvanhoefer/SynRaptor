"""Combinations of multiple drugs are defined here.
"""
import numpy as np
import scipy as sp
from SynRaptor import drug
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt




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
            prod(single_response_i). If gradient is True the gradient will be returned as well.

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
                the gradient
            """
            self._check_bliss_requirements()
            l = len(self.drug_list)

            # For monotone increasing drugs the Bliss response is 1-prod(1-y_i)
            if self.drug_list[0].monotone_increasing:
                prod = 1
                for i in range(l):
                    prod *= 1 - self.drug_list[i].get_response(dose_combination[i])
                if not gradient:
                    return 1 - prod
                # Here the gradient is calculated
                grad = np.nan * np.ones((l, 3))
                responses = np.nan * np.ones(l)
                for i in range(l):
                    (responses[i], grad[i]) = self.drug_list[i].get_response(dose_combination[i],
                                                                             self.drug_list[i].parameters, True)
                    grad[i] = prod / (1 - responses[i]) * grad[i]
                grad = self.matrix_to_vector(grad)  # now gradient looks like [a0 n0 s0 a1 n1 s1 ...]
                return 1 - prod, grad
            # For monotone decreasing drugs the Bliss response is prod(y_i)
            else:
                prod = 1
                for i in range(l):
                    prod *= self.drug_list[i].get_response(dose_combination[i])
                if not gradient:
                    return prod
                # Here the gradient is calculated
                grad = np.nan * np.ones((l, 3))
                responses = np.nan * np.ones(l)
                for i in range(l):
                    (responses[i], grad[i]) = self.drug_list[i].get_response(dose_combination[i],
                                                                             self.drug_list[i].parameters, True)
                    grad[i] = prod / responses[i] * grad[i]
                grad = self.matrix_to_vector(grad) #now gradient looks like [a0 n0 s0 a1 n1 s1 ...]
                return prod, grad




    def _check_bliss_requirements(self):
        """
        Checks if requirements for get_bliss_response are fulfilled.

        Checks if for monotone increasing drugs the control_response is 0, and the maximal effect is <= 1. As the
        control response is the same for all drugs it it sufficient to check for drug_list[0].
        For decreasing drugs the control_response has to be 1 and the parameter s of Hill curve <=1. If one requirement
        is not met, False is returned. Otherwise True is returned.
        """
        for i in range(len(self.drug_list)):
            if self.drug_list[i].parameters[2] > 1:
                raise RuntimeError('parameter s should not be larger than 1')
        if self.drug_list[0].monotone_increasing:
            if not (self.drug_list[0].control_response == 0):
                raise RuntimeError('control response should be 0')
        else:
            if not (self.drug_list[0].control_response == 1):
                raise RuntimeError('control response should be 1')
        return


    def get_multiple_bliss_responses(self,
                                     dose_combinations: np.array,#2dimensional
                                     gradient: bool):
        """
        Computes multiple Bliss responses using get_bliss_response. Also returns the gradient if wanted.

        Parameters
        ----------
        dose_combinations: np.array
            two dimensional array with dose_combinations[i] containing a dose or each drug in self.drug_list

        gradient: bool
            determines wether gradients should be returned as well

        Returns
        -------
        responses: np.array
            responses according to Bliss

        grad: np.array
            two dimensional array so that the ith coloumn corresponds to gradient of bliss at dose_combinations[i]
        """
        l = len(dose_combinations)
        responses = np.nan * np.ones(l)
        if not gradient:
            for i in range(l):
                responses[i] = self.get_bliss_response(dose_combinations[i], False)
            return responses
        l2 = len(dose_combinations[0])
        grad = np.nan * np.ones((l, 3 * l2))
        for i in range(l):
            (responses[i], grad[i]) = self.get_bliss_response(dose_combinations[i], True)#row of grad looks like [dda0 ddn0 dds0 dda1 ddn1 dds1 ...].]
        grad = np.transpose(grad)#coloumn looks like [dda0 ddn0 dds0 dda1 ddn1 dds1 ...]
        return responses, grad


    def evaluate_log_likelihood(self,
                                responses: np.array,
                                dose_combinations, #doses[i] is a dose_combination, so doses is a 2 dim array
                                gradient: bool):
        """
        Calculates -2*log-likelihood (or the RSS). Also calculates gradient if wanted.

        Parameters
        ----------
        responses: np.array
            measured responses of drug experiment

        dose_combinations: np.array
            two dimensional array with dose_combinations[i] containing a dose for each drug in self.drug_list

        gradient: bool
            determines wether gradients should be returned as well

        Returns
        -------
        sum / sigma2: float
            the log-likelihood

        grad: np.array
            the gradient of log-likelihood containing partial derivatives for parameters

        """
        sigma2 = 1 #TODO implement get_sigma
        sum = 0
        l = len(responses)
        if not gradient:
            prediction = self.get_multiple_bliss_responses(dose_combinations, False)
            for i in range(l):
                sum += (responses[i] - prediction[i])**2
            return sum / sigma2

        (prediction, prediction_grad) = self.get_multiple_bliss_responses(dose_combinations, True)
        for i in range(l):
            sum += (responses[i] - prediction[i]) ** 2
        grad = np.dot(prediction_grad, 2 * (prediction - responses))#grad looks like [dda0 ddn0 dds0 dda1 ddn1 dds1 ...]
        return sum / sigma2, grad





    def evaluate_validation_residual(self,
                                     validation_response,
                                     validation_dose: np.array,
                                     gradient: bool):
        """
        Calculates the squared residual of validation data point. Also returns the gradient if wanted.

        Parameters
        ----------
        validation_response: float
            measured response of validation experiment

        validation_dose: np.array
            dose of validation experiment

        gradient: bool
            determines wether gradient should be returned as well

        Returns
        -------
        residual: float
            the calculated squared residual of validation experiment

        grad: np.array
            the gradient of sqaured residual of validation experiment containing partial derivatives for parameters

        """
        sigma2 = 1 #TODO use get_sigma
        if not gradient:
            return ((validation_response - self.get_bliss_response(validation_dose, False)) / sigma2) ** 2
        (response, grad) = self.get_bliss_response(validation_dose, True)
        residual = ((validation_response - response) / sigma2) ** 2
        grad = 2 * grad * (response - validation_response)
        return residual, grad


    def fit_to_old_data(self,
                         responses: np.array,
                         doses): #doses[i] is a dose_combination, so doses is a 2 dim array
        """
        Fits drug parameters to data.

        Parameters
        ----------
        responses: np.array
            measured responses

        doses: np.array
            two dimensional array where dose[i] containes dose for each drug in drug_list

        Returns
        -------
        minimum_value: float
            minimal value of -2LL for given doses and responses

        minimum_parameters: np.array
            optimized parameters that minimize -2LL
        """


        def min2loglikelihood(parameters: np.array): #parameters is array of length 3 * len(drug_list)
            self.parameters_to_drug_list(parameters)
            return self.evaluate_log_likelihood(responses, doses, False)

        l = 3 * len(self.drug_list)
        bounds = np.ones((l), dtype=(float, 2))
        for i in range(l):
            bounds[i] = 1e-8, 0.99  # for Bliss s has to be between 0 and 1

        initialParameters = self._get_optimization_starts(bounds, 10)
        minimum_value = float('inf')
        minimum_parameters = None
        for i in range(10):
            solution = minimize(min2loglikelihood, initialParameters[i], method='TNC', jac=False, bounds=bounds)
            if solution.fun < minimum_value:
                minimum_value = solution.fun
                minimum_parameters = solution.x
        self.parameters_to_drug_list(minimum_parameters)
        return minimum_value, minimum_parameters

    def fit_to_full_data(self,
                         validation_response: float,
                         validation_dose: np.array,
                         responses: np.array,
                         doses): #doses[i] is a dose_combination, so doses is a 2 dim array
        """
        Fits drug parameters to data including validation experiment.

        Parameters
        ----------
        validation_dose: float
            response data of validation experiment

        validation_dose: np.array
            dose_combination for validation experiment

        responses: np.array
            measured responses before validation experiment

        doses: np.array
            two dimensional array where dose[i] contains dose for each drug in drug_list

        Returns
        -------
        minimum_value: float
              minimal value of -2LL for given data

        minimum_parameters: np.array
             optimized parameters that minimize -2LL
        """

        def min2loglikelihood(parameters: np.array): #parameters is array of length 3 * len(drug_list)
            self.parameters_to_drug_list(parameters)
            (r1, grad1) = self.evaluate_log_likelihood(responses, doses, True)
            (r2, grad2) = self.evaluate_validation_residual(validation_response, validation_dose, True)
            return r1 + r2, grad1 + grad2


        l = 3 * len(self.drug_list)
        bounds = np.ones((l), dtype=(float,2))
        for i in range(l):
            bounds[i] = 1e-8, 0.99 # for Bliss s has to be between 0 and 1

        initialParameters = self._get_optimization_starts(bounds, 10)
        minimum_value = float('inf')
        minimum_parameters = None
        for i in range(10):
            solution = minimize(min2loglikelihood, initialParameters[i], method='TNC', jac=True, bounds=bounds)
            if solution.fun < minimum_value:
                minimum_value = solution.fun
                minimum_parameters = solution.x
        self.parameters_to_drug_list(minimum_parameters)
        return minimum_value, minimum_parameters


    def _get_optimization_starts(self,
                                 bounds: np.array,
                                 n_starts: int):
        """
        Computes initial_values for optimization using Latin Hypercube sampling.

        Parameters
        ----------
        bounds: np.array
            parameter bounds for optimization

        n_starts: int
            number of optimization starts

        Returns
        -------
        np.transpose(initial_values): np.array
            initial_values for optimization
        """
        l = len(bounds)
        initial_values = [[0] * n_starts for i in range(l)]
        for i in range(l):
            perm = np.random.permutation(n_starts)
            initial_values[i] = np.array([bounds[i][0] + (bounds[i][1] - bounds[i][0]) /\
                                          n_starts * (perm[j] + 0.5) for j in range(n_starts)])
        return np.transpose(initial_values)

    def matrix_to_vector(self,
                         matrix: np.array):
        """
        Reshapes two dim array into one dim array.
        """
        l1 = len(matrix)
        l2 = len(matrix[0])
        vector = np.nan * np.ones(l1*l2)
        for i in range(l1):
            for j in range(l2):
                vector[l2 * i + j] = matrix[i][j]
        return vector


    def parameters_to_drug_list(self,
                                parameters: np.array):
        """
        Saves array of parameters in self.drug_list.
        """
        for i in range(len(self.drug_list)):
            self.drug_list[i].parameters[0] = parameters[3 * i]
            self.drug_list[i].parameters[1] = parameters[3 * i + 1]
            self.drug_list[i].parameters[2] = parameters[3 * i + 2]
        return


    def drug_list_to_parameters(self):
        """
        Creates array containing the parameters of self.drug_list.
        """
        l = len(self.drug_list)
        parameters = np.nan * np.ones(3*l)
        for i in range(l):
            parameters[3 * i] = self.drug_list[i].parameters[0]
            parameters[3 * i + 1] = self.drug_list[i].parameters[1]
            parameters[3 * i + 2] = self.drug_list[i].parameters[2]
        return parameters


    def get_hand_response(self,
                          dose_combination: np.array,
                          gradient: bool):
            """

            Compute the hand response
            """


            def f(y, t,
                  dose_combination: np.array,
                  ):
                l = len(self.drug_list)
                s = sum(dose_combination)
                r = 0
                for i in range(l):
                    dev = self.drug_list[i].get_derivative(self.drug_list[i].inverse_evaluate(y))
                    r += (dose_combination[i] / s) * dev
                return r
            # initial condition
            y0 = [1e-7, 1e-7]
            # timepoints
            t = np.linspace(1e-7,1)
            # solve ode
            y = odeint(f,y0,t, args = (dose_combination,))
            # wir wollen y[-1] also das letzte element im array
            return y[-1]


    def get_hsa_response(self,
                         dose_combination: np.array,
                         gradient: bool):
            """

            Compute the HSA response

            This function calculates the HSA response, taking into consideration wether the single dose effect curves are
            monotone increasing or not. For monotone increasing drugs the maximal response is returned. For monotone
            decreasing drugs the minimal response of the single drugs is returned. The gradient regarding the parameters
             a, n and s of every single drug can be returned as well.

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
                gradient (partial derivatives)
            """
            responses = [drug_list[i].get_response(dose_combination[i]) for i in range(len(self.drug_list))]
            if gradient:
                if drug_list[0].monotone_increasing:
                    # monotone increasing and gradient
                    l = len(self.drug_list)
                    max = np.argmax(responses)
                    (response, gradm) = self.drug_list[max].get_response(dose_combination[max],
                                                                         self.drug_list[max].parameters, True)
                    grad = np.zeros((l, 3))
                    grad[max] = gradm
                    return response, grad
                else:
                    # monotone decreasing and gradient
                    l = len(self.drug_list)
                    min = np.argmin(responses)
                    (response, gradm) = self.drug_list[min].get_response(dose_combination[min],
                                                                         self.drug_list[min].parameters, True)
                    grad = np.zeros((l, 3))
                    grad[min] = gradm
                    return response, grad
            else:
                if drug_list[0].monotone_increasing:
                    # monotone increasing without gradient
                    return np.max(responses)
                else:
                    # monotone decreasing without gradient
                    return np.min(responses)


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


#This code may be used for testing.
print('hallo')
x = np.array([1,2,3,4,5])
y = np.array([2,4,6,6,7])
z = np.array([1,2,4,5,7])


A = drug.Drug(x, 0.0001 * y, False, 1)
A.fit_parameters(10)

#print(A.get_derivative(7))
#print(A.inverse_evaluate(7))

B = drug.Drug(y, 0.0001 * z, False, 1)
B.fit_parameters(10)


C = drug.Drug(x, 0.0001 * z, False, 1)
C.fit_parameters(10)

D = drug.Drug(y, 0.0001 * y, False, 1)
D.fit_parameters(10)


A.parameters[2] = 0.4
B.parameters[2] = 0.3
C.parameters[2] = 0.7
D.parameters[2] = 0.2

drug_list = [A,B,C,D]
Comb = Combination(drug_list)
dose1 = np.array([0.5,0.5,0.5,0.8])
dose2 = np.array([0.6,0.5,0.5,0.2])
dose3 = np.array([0.5,0.3,0.5,0.1])
dosez = np.array([0.5,0.5,0.4,0.6])
doses = np.array([dose1,dose2, dose3])


responses = np.array([0.6,0.8, 0.7])
#res = Comb.get_hsa_response(dose, True)

#res2 = Comb.get_bliss_response(dosez, True)
#res2 = Comb.get_multiple_bliss_responses(doses, True)
#res4 = Comb.evaluate_log_likelihood(responses,doses, True)
#res3 = Comb.get_hand_response(dose,True)
#print(res)
#print(res4)
#print(res3)


#c = Comb.fit_to_full_data(0.7, dosez,responses,doses)
#d = Comb.fit_to_old_data(responses, doses)
#print(c)
#print(d)
#p = Comb.drug_list_to_parameters()
#print(p)
#Comb.parameters_to_drug_list(p)
#a = Comb.evaluate_validation_residual(0.7,dosez)
#b = Comb.evaluate_log_likelihood(responses,doses)
#print(a)
#print(b)
