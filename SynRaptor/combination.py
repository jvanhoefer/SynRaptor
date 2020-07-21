"""Combinations of multiple drugs are defined here.
"""
import numpy as np
import scipy as sp
from SynRaptor import drug
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import chi2

import math
import numpy.matlib


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

        self._set_sigma2()

    def get_loewe_response(self,
                           dose_combination: np.array,
                           gradient: bool,
                           parameters: np.array = None):
        """
        Calculates the Loewe response and gradient if required.

        The Loewe response is calculated through bisection.
        Uses implicit functions theorem to calculate the gradient if required. THe theorem states that:
        dy_dx_i = - d loewe_equation_dx_i / d loewe_equation_dy. To save code these derviatives are calculated through
        the use of dy_da_i, dy_dn_i and dy_ds_i functions. To calculate d loewe_equation_dy the di_dy function is used
        several times.

        Parameters
        ----------
        dose_combination: np.array
                gives the doses for the drugs that will be combined

        gradient: bool
            determines whether gradient should be returned as well

        parameters: np.array = None
            parameters of drugs (if None parameters of self.drug_list are used)


        Returns
        -------
        effect: float
            the response of the combined drug

        grad: np.array
            the gradient
        """

        number_of_drugs = len(dose_combination)

        if parameters is None:
            parameters = [drug.parameters for drug in self.drug_list]

        def loewe_equation(effect):
            s = sum(dose_combination[i] / self.drug_list[i].inverse_evaluate(effect, parameters[i]) for i in
                    range(number_of_drugs))
            return s - 1

        effect = sp.optimize.bisect(loewe_equation, self.drug_list[0].control_response, self.drug_list[0].parameters[2])

        if not gradient:
            return effect

        def get_parameters(i: int):
            return self.drug_list[i].control_response, parameters[i][0], parameters[i][1], parameters[i][2]

        def di_dy(y,
                  i: int):  # index of drug in drug_list
            """
            Calculates the derivative of 1/f_i^{-1}(y)
            """
            w, a, n, s = get_parameters(i)
            return - s * a ** n * (a ** n * (y - w) / (s + w - y)) ** (-1 / (n - 1)) / (n * (s + w - y) ** 2)

        divFy = sum(dose_combination[i] * di_dy(effect, i) for i in range(number_of_drugs))  # calculates dF/dy

        def dy_da_i(y,
                    i: int):
            """
            Calculates dy/da_i
            """
            w, a, n, s = get_parameters(i)
            return dose_combination[i] * (a ** n * (y - w) / (s + w - y)) ** (-1 / n) / a / divFy

        def dy_dn_i(y,
                    i: int):
            """
            Calculates dy/dn_i
            """
            w, a, n, s = get_parameters(i)
            return dose_combination[i] * ((y - w) / (s + w - y)) ** (-1 / n) * math.log((y - w) / (s + w - y)) / (
                    a * n ** 2) / divFy  # a,n,w,s,y are positive

        def dy_ds_i(y,
                    i: int):
            """
            Calculates dy/ds_i
            """
            w, a, n, s = get_parameters(i)
            return dose_combination[i] * (a ** n * (y - w) / (s + w - y)) ** (-1 / n) / (n * (s + w - y)) / divFy

        grad = [np.array([dy_da_i(effect, i), dy_dn_i(effect, i), dy_ds_i(effect, i)]) for i in range(number_of_drugs)]

        return effect, self._matrix_to_vector(grad)

    def get_bliss_response(self,
                           dose_combination: np.array,
                           gradient: bool,
                           parameters: np.array = None):
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

        parameters: np.array = None
            parameters of drugs (if None parameters of self.drug_list are used)


        Returns
        -------
        response: float
            the response of the combined drug

        grad: np.array
            the gradient
        """

        l = len(dose_combination)
        if parameters is None:
            parameters = [drug.parameters for drug in self.drug_list]
        self._check_bliss_requirements(parameters)

        # For monotone increasing drugs the Bliss response is 1-prod(1-y_i)
        if self.drug_list[0].monotone_increasing:
            if not gradient:
                return 1 - np.prod(
                    [1 - self.drug_list[i].get_response(dose_combination[i], parameters[i]) for i in range(l)])

            response_grad_matrix = [self.drug_list[i].get_response(dose_combination[i], parameters[i], True) for i in range(l)]
            oneminusresponses = [1 - response_grad_matrix[i][0] for i in range(l)]
            prod = np.prod(oneminusresponses)
            grad = [prod / (oneminusresponses[i]) * response_grad_matrix[i][1] for i in range(l)]
            grad = self._matrix_to_vector(grad)  # now gradient looks like [a0 n0 s0 a1 n1 s1 ...]
            return 1 - prod, grad
        # For monotone decreasing drugs the Bliss response is prod(y_i)
        else:
            if not gradient:
                return np.prod([self.drug_list[i].get_response(dose_combination[i], parameters[i]) for i in range(l)])

            response_grad_matrix = [self.drug_list[i].get_response(dose_combination[i], parameters[i], True) for i in range(l)]
            responses = [response_grad_matrix[i][0] for i in range(l)]
            prod = np.prod(responses)
            grad = [prod / responses[i] * response_grad_matrix[i][1] for i in range(l)]
            grad = self._matrix_to_vector(grad)  # now gradient looks like [a0 n0 s0 a1 n1 s1 ...]
            return prod, grad

    def _check_bliss_requirements(self,
                                  parameters: np.array):
        """
        Checks if requirements for get_bliss_response are fulfilled.

        Checks if for monotone increasing drugs the control_response is 0, and the maximal effect is <= 1. As the
        control response is the same for all drugs it it sufficient to check for drug_list[0].
        For decreasing drugs the control_response has to be 1 and the parameter s of Hill curve <=1. If one requirement
        is not met, False is returned. Otherwise True is returned.
        """
        for i in range(len(self.drug_list)):
            if parameters[i][2] > 1:
                raise RuntimeError('In bliss model parameter s should not be larger than 1')
        if self.drug_list[0].monotone_increasing:
            if not (self.drug_list[0].control_response == 0):
                raise RuntimeError('For monotone increasing drugs in bliss model control response should be 0')
        else:
            if not (self.drug_list[0].control_response == 1):
                raise RuntimeError('Für monotone decreasing drugs in Bliss model control response should be 1')
        return

    def get_hand_response(self,
                          dose_combination: np.array,
                          gradient: bool,
                          parameters: np.array = None):
        """

        Compute the hand response.

        Uses odeint to solve hand ODE and finite differences to calculate gradient.

        Parameters
        ----------
        dose_combination: np.array
            gives the doses for the drugs that will be combined

        gradient: bool
            determines wether gradient should be returned as well

        parameters: np.array = None
            parameters of drugs (if None parameters of self.drug_list are used)


        Returns
        -------
        y[-1]: float
            the response of the combined drug

        grad: np.array
            the gradient
        """
        if parameters is None:
            parameters = [drug.parameters for drug in self.drug_list]

        s = sum(dose_combination)

        def f(y, t,
              dose_combination: np.array,
              ):
            """
            right hand side of hand ODE
            """
            number_of_drugs = len(self.drug_list)

            r = 0
            for i in range(number_of_drugs):
                drug_sensivity = self.drug_list[i].get_derivative(self.drug_list[i].inverse_evaluate(y, parameters), parameters)
                r += (dose_combination[i] / s) * drug_sensivity
            return r

        # initial condition
        y0 = 1e-7
        # timepoints
        t = np.array([0, s])
        # solve ode
        y = odeint(f, y0, t, args=(dose_combination,))
        if not gradient:
            return y[-1] #returns last element in array
        """
        gradient using finite differences
        """

        def fv(effect, t,
               dose_combination: np.array,
               i: int,
               v: np.array,
               ):
            """
            ith summand of right hand side of hand ODE with changed parameters for finite differences
            """

            p = np.array([parameters[0] + v[0], parameters[1] + v[1], parameters[2] + v[2]])
            drug_sensivity = self.drug_list[i].get_derivative(self.drug_list[i].inverse_evaluate(effect, p), p)

            return (dose_combination[i] / s) * drug_sensivity

        grad = np.array([])
        for i in range(len(self.drug_list)):
            for j in range(3):
                v = np.array([0, 0, 0])
                v[j] = v[j] + 0.1

                y1 = odeint(fv, y0, t, args=(dose_combination, i, v))
                y2 = odeint(fv, y0, t, args=(dose_combination, i, -v))
                grad = np.append(grad, (y1[-1] - y2[-1]) / 0.2)
                print(y1[-1], y2[-1])
        return y[-1], grad

    def get_hsa_response(self,
                         dose_combination: np.array,
                         gradient: bool,
                         parameters: np.array = None):
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

        parameters: np.array
            parameters for single drugs

        Returns
        -------
        response: float
            the response of the combined drug

        grad: np.array
            gradient (partial derivatives)
        """
        l = len(self.drug_list)
        if parameters is None:
            parameters = [self.drug_list[i].parameters for i in range(l)]

        responses = [self.drug_list[i].get_response(dose_combination[i]) for i in range(l)]

        if self.drug_list[0].monotone_increasing:
            hsa = np.argmax(responses)
        else:
            hsa = np.argmin(responses)

        if gradient:
            (response, gradm) = self.drug_list[hsa].get_response(dose_combination[hsa], parameters[hsa], True)
            grad = np.zeros((l, 3))
            grad[hsa] = gradm
            return response, self._matrix_to_vector(grad)
        else:
            return responses[hsa]


    def evaluate_log_likelihood_single_drug_data(self,
                                                 parameters: np.array,  # 2dim matrix
                                                 gradient: bool):
        """
        Evaluates -2 * log likelihood of data without validation data.

        Parameters
        ----------
        parameters: np.array
            parameters of drugs

        gradient: bool
            determines wether gradient should be returned as well

        Returns
        -------
        loglikelihood: float

        grad: np.array
            gradient of loglikelihood

        """
        if not gradient:
            return np.sum([drug.evaluate_lsq_residual(drug.parameters, False)
                           for drug in self.drug_list])

        else:
            number_of_drugs = len(self.drug_list)
            (sum_of_residuals, grad) = self.drug_list[0].evaluate_lsq_residual(parameters[0], True)

            for i in range(1, number_of_drugs):
                (lsq_new, grad_new) = self.drug_list[i].evaluate_lsq_residual(parameters[i], True)
                sum_of_residuals += lsq_new
                grad = np.append(grad, grad_new)

            return sum_of_residuals / self.sigma2, grad / self.sigma2


    def evaluate_validation_residual(self,
                                     validation_response,
                                     validation_dose: np.array,
                                     gradient: bool,
                                     parameters: np.array = None,  # 2dim array
                                     null_model: str = 'bliss'):
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

        parameters: np.array = None
            parameters for drugs

        null_model: str
            null_model that is used

        Returns
        -------
        residual: float
            the calculated squared residual of validation experiment

        grad: np.array
            the gradient of sqaured residual of validation experiment containing partial derivatives for parameters

        """
        if null_model == 'bliss':
            get_combination_response = self.get_bliss_response
        elif null_model == 'hsa':
            get_combination_response = self.get_hsa_response
        elif null_model == 'loewe':
            get_combination_response = self.get_loewe_response
        elif null_model == 'hand':
            get_combination_response = self.get_hand_response
        else:
            ValueError("invalid null model")

        if parameters is None:
            parameters = [drug.parameters for drug in self.drug_list]

        if not gradient:
            return ((validation_response - get_combination_response(validation_dose, False,
                                                                    parameters)) / self.sigma2) ** 2
        (response, grad) = get_combination_response(validation_dose, True, parameters)
        residual = ((validation_response - response) / self.sigma2) ** 2
        grad = 2 * grad * (response - validation_response)
        return residual, grad

    def fit_to_full_data(self,
                         validation_response: float,
                         validation_dose: np.array,
                         null_model: str = 'bliss'):
        """
        Fits drug parameters to data including validation experiment.

        Parameters
        ----------
        validation_dose: float
            response data of validation experiment

        validation_dose: np.array
            dose_combination for validation experiment

        null_model: str
            null model that is used

        Returns
        -------
        solution.fun: float
              minimal value of -2LL for given data
        """

        def min2loglikelihood(parameters: np.array,  # parameters is array of length 3 * len(drug_list)
                              null_model: str):
            parameters = self._vector_to_matrix(parameters)
            (r1, grad1) = self.evaluate_log_likelihood_single_drug_data(parameters, True)
            (r2, grad2) = self.evaluate_validation_residual(validation_response, validation_dose, True, parameters,
                                                            null_model)
            return r1 + r2, grad1 + grad2


        bounds = numpy.matlib.repmat(np.array([(1e-8, 10), (1e-8, 10), (1e-8, 0.99)]), len(self.drug_list), 1)


        initial_parameters = self._drug_list_to_parameters()
        solution = minimize(min2loglikelihood, initial_parameters, args=null_model, method='TNC', jac=True,
                            bounds=bounds)
        return solution.fun


    def fit_to_old_data(self):
        """
        Fits drug parameters to data including validation experiment.

        Parameters
        ----------
        no inputs

        Returns
        -------
        solution.fun: float
              minimal value of -2LL for given data

        solution.x: np.array
             optimized parameters that minimize -2LL
        """

        def min2loglikelihood(parameters: np.array):  # parameters is array of length 3 * len(drug_list)
            parameters = self._vector_to_matrix(parameters)
            (r, grad) = self.evaluate_log_likelihood_single_drug_data(parameters, True)
            return r , grad


        bounds = numpy.matlib.repmat(np.array([(1e-8, 10), (1e-8, 10), (1e-8, 0.99)]), len(self.drug_list), 1)


        initial_parameters = self._drug_list_to_parameters()
        solution = minimize(min2loglikelihood, initial_parameters, method='TNC', jac=True,
                            bounds=bounds)
        return solution.fun, solution.x


    def _matrix_to_vector(self,
                          matrix: np.array):
        """
        Reshapes two dim array into one dim array.
        """
        return np.reshape(matrix, -1, order='C')

    def _vector_to_matrix(self,
                          vector: np.array):
        """
        Reshapes vector to two dim array with len(matrix[i])=3.
        """
        return np.reshape(vector, (-1, 3), order='C')

    def _drug_list_to_parameters(self):
        """
        Creates array containing the parameters of self.drug_list.
        """
        number_of_drugs = len(self.drug_list)
        parameters = np.nan * np.ones(3 * number_of_drugs)
        for i in range(number_of_drugs):
            parameters[3 * i: 3 * i + 3] = self.drug_list[i].parameters
        return parameters


    def get_significance(self,
                               dose_combination: np.array,
                               responses: np.array,
                               null_model: str = 'bliss'):#TODO at the moment only for one response
        """
        Compute the significance level for a given dose and measurements and null_model.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        chi2.cdf is used to calculate significance.

        Parameters
        ----------
        dose_combination: np.array
            dose of validation experiments

        responses: np.array
            responses of validation experiment

        null_model: str = 'bliss'
            null model to calculate significance for

        Returns
        -------
        chi2: float
            significance level
        """

        (rss_y_y, theta_y) = self.fit_to_old_data()
        rss_y_yz = rss_y_y + self.evaluate_validation_residual(responses, dose_combination, False,
                                                     self._vector_to_matrix(theta_y), null_model)
        rss_yz_yz = self.fit_to_full_data(responses, dose_combination, null_model)
        difference = rss_y_yz - rss_yz_yz
        return chi2.cdf(difference, 1, loc=0, scale=1)


    def _check_drug_consistency(self,
                                drug_list: list) -> bool:
        """
        check, if all drugs are either mon incr or decr. ...
        check, if all drugs share the same w_0 ...
        If yes return True, if not return False.
        """
        l = len(drug_list)
        control_response = drug_list[l - 1].control_response
        mon = drug_list[l - 1].monotone_increasing
        for i in range(l - 1):
            if not (drug_list[i].control_response == control_response):
                return False
            if not (drug_list[i].monotone_increasing == mon):
                return False
        return True

    def _set_sigma2(self):  # TODO numerisch super instabil
        """
        Compute the (optimal) sigma for the given drugs in drug_list
        #formel von hiererchischer optimierung für alle drugs zusammen
        """
        l = len(self.drug_list)
        sum = 0
        for i in range(l):
            sum += self.drug_list[i].get_sigma2()
        self.sigma2 = sum / l
        return


# This code may be used for testing.
# print('hallo')
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 6, 7])
z = np.array([1, 2, 4, 5, 7])

A = drug.Drug(x, 0.0001 * y, True, 0)
A.fit_parameters(10)

# print(A.get_derivative(7))
# print(A.inverse_evaluate(7))

B = drug.Drug(y, 0.0001 * z, True, 0)
B.fit_parameters(10)

C = drug.Drug(x, 0.0001 * z, True, 0)
C.fit_parameters(10)

D = drug.Drug(y, 0.0001 * y, True, 0)
D.fit_parameters(10)

A.parameters = np.array([3, 2, 0.8])
B.parameters = np.array([7, 1, 0.8])
C.parameters = np.array([3, 1.5, 0.4])
D.parameters = np.array([4, 2, 1])

drug_list = [A, B]
Comb = Combination(drug_list)

dose1 = np.array([3, 7])
response1 = np.array([0.1, 0.7])

print(Comb.get_significance(dose1, 0.1, 'bliss'))

