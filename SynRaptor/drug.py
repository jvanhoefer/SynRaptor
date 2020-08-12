""" Single Drugs are defined here.
"""
import numpy as np
import scipy as sp
import math
from scipy.optimize import minimize


class Drug:
    """
    Drug stores a parametric representation of a dose response curve
    (=Hill curve) together with dose response data.

    The hill curve is parametrized as

                        w_0 + s*x^n /(a^n+x^n)

    Attributes
    ----------

    parameters: np.array
        [a, n, s]
        a: float
            Half-Max
        n: float
            Hill-Coefficient
        s: float
            maximal effect

    dose_data: np.array
        doses from the dose-response data

    response_data: np.array
        responses from the dose-response data

    monotone_increasing: bool
        flag, indicating if the curve is monotone increasing

    control_response: float
        response for zero dose (in the upper formula: w_0)

    Methods
    -------

    Drug: Drug
        Constructor

    get_response:
        calculates response for given parameters and single dose and returns gradient if requested

    get_multiple_responses:
        calculates responses for given parameters and multiple doses and returns gradients if requested

    evaluate_lsq_residual:
        Evaluates the LSQ residual for given parameters and returns gradient if requested

    fit_parameters:
        fits parameters to data and stores new parameters in drug

    _get_optimizations_starts:
        samples initial values in the bounds via Latin Hypercube sampling

    _set_dose_and_response:
        sets dose and response data. Checks dimensions

    """

    def __init__(self,
                 dose_data: np.array = None,
                 response_data: np.array = None,
                 monotone_increasing: bool = False,
                 control_response: float = 1):
        """
        Constructor
        """

        self.parameters = None

        self.monotone_increasing = monotone_increasing
        self.control_response = control_response

        if dose_data is not None and response_data is not None:
            self._set_dose_and_response(dose_data, response_data)
            # self._set_monotony()

    def get_response(self,
                     dose: float,
                     parameters: np.array = None,
                     gradient: bool = False):
        """ calculates response for given parameters and single dose and returns gradient if requested

        If parameters are not given, self.parameters of the drug will be used.

            Parameters
            ----------
            dose: float
                dose of the given drug

            parameters: np.array
                parameters a, n and s of the hill curve

            gradient: bool
                decides whether gradient should be returned as well

            Returns
            -------
            response_value: float
                by hill curve predicted response for drug and dose

            grad: np.array
                gradient of hill curve at dose dose

               """

        if parameters is None:
            parameters = self.parameters
        monotone_increasing = self.monotone_increasing
        control_response = self.control_response

        a = parameters[0]
        n = parameters[1]
        s = parameters[2]
        if monotone_increasing:
            response_value: float = control_response + s * dose ** n / (a ** n + dose ** n)
        else:
            response_value: float = control_response - s * dose ** n / (a ** n + dose ** n)
        if not gradient:
            return response_value
        if dose == 0:
            return response_value, np.array([0.001, 0.001, 0.001])
        grad = np.array([np.nan, np.nan, np.nan])
        grad[0] = -s * dose ** n * a ** (n - 1) * n / ((a ** n + dose ** n) ** 2)
        grad[1] = a ** n * s * dose ** n * math.log(dose / a) / ((a ** n + dose ** n) ** 2)
        grad[2] = dose ** n / (a ** n + dose ** n)
        if not monotone_increasing:
            grad = -grad
        return response_value, grad

    def get_derivative(self,
                       x: float,
                       parameters=None):
        if parameters is None:
            parameters = self.parameters
        monotone_increasing = self.monotone_increasing
        a = parameters[0]
        n = parameters[1]
        s = parameters[2]
        if monotone_increasing:
            derivative: float = n * s * a ** n * x ** (n - 1) / (a ** n + x ** n) ** 2
        else:
            derivative: float = - n * s * a ** n * x ** (n - 1) / (a ** n + x ** n) ** 2
        return derivative

    def inverse_evaluate(self,
                         effect: float,
                         parameters: np.array = None):
        if parameters is None:
            parameters = self.parameters

        # for increasing
        if self.monotone_increasing:
            if effect < self.control_response:
                return float('-inf')
            elif effect > self.control_response + parameters[2]:
                return float('inf')

            return ((effect - self.control_response) * parameters[0] ** parameters[1] /
                    (parameters[2] - effect + self.control_response)) ** (1 / parameters[1])
        # for decreasing
        else:
            if effect > self.control_response:
                return float('-inf')
            elif effect < self.control_response - parameters[2]:
                return float('inf')
            """
            if (effect + self.control_response) * parameters[0] ** parameters[1] / \
                    (parameters[2] - effect - self.control_response) < 0:
                print('can not calculate fractional power of negative float in inverse evaluate')
                return 0
            """
            return ((self.control_response - effect) /
                    (parameters[2] + effect - self.control_response)) ** (1 / parameters[1]) * parameters[0]

    def sensitivity(self,
                    effect: float,
                    parameters: np.array = None):
        if parameters is None:
            parameters = self.parameters

        if effect > self.control_response:
            return 0
        elif effect < self.control_response - parameters[2]:
            return 0

        return self.get_derivative(self.inverse_evaluate(effect, parameters), parameters)

    def get_multiple_responses(self,
                               doses: np.array,
                               parameters: np.array = None,
                               gradient: bool = False):

        """calculates responses for given parameters and multiple doses and returns gradients if requested

        Uses get_response to calculate results for each dose.

        Parameters
        ---------

        doses: np.array
            doses for which results shall be calculated

        parameters: np.array
                parameters a, n and s of the hill curve

        gradient: bool
                decides wether gradient should be returned

        Returns
        -------

        responses: np.array
            responses predicted by hill curve for doses doses

        grad: np.array
            gradients of hill curve at doses doses

          """
        l = len(doses)
        responses = np.nan * np.ones(l)
        if not gradient:
            for i in range(l):
                responses[i] = self.get_response(doses[i], parameters, False)
            return responses
        grad = np.nan * np.ones((l, 3))
        for i in range(l):
            (responses[i], grad[i]) = self.get_response(doses[i], parameters, True)
        grad = np.transpose(grad)
        return responses, grad

    def evaluate_lsq_residual(self,
                              parameters: np.array,
                              gradient: bool = False):
        """ Evaluates the LSQ residual for given parameters AND returns gradient if requested

        Uses responses and grad of get_multiple_responses to calculate the lsq residual and its gradient (if requested)
        for hill curve parameters

        Parameters
        ----------

        parameters: np.array
            hill curve parameters

        gradient: bool
            decides whether gradient shall be calculated

        Returns
        -------

        lsq_residual: float
            lsq_residual of hill curve and Drug

        grad: np.array
            gradient of lsq_residual regarding hill curve parameters for doses self.doses

        """
        lsq_residual = 0
        if not gradient:
            responses = self.get_multiple_responses(self.dose_data, parameters)
            for i in range(len(self.response_data)):
                lsq_residual += (self.response_data[i] - responses[i]) ** 2
            lsq_residual = lsq_residual
            return lsq_residual

        (responses, responses_grad) = self.get_multiple_responses(self.dose_data, parameters, True)
        for i in range(len(self.response_data)):
            lsq_residual += (self.response_data[i] - responses[i]) ** 2
        lsq_residual = lsq_residual
        grad = np.dot(responses_grad, 2 * (responses - self.response_data))
        return lsq_residual, grad

    def fit_parameters(self,
                       n_starts: int = 10):
        """ fits parameters to data and stores new parameters in drug

        Uses minimize (scipy) to minimize evaluate_lsq_residual using its gradient as well. Iterates over n_starts by
        _get_optimization_starts sampled initial_values and returns and stores the parameters of the minimal result.

        Parameters
        ----------

        n_starts: int
            number of initial_values for minimization

        Returns
        -------

        minimum_parameters: np.array
            parameters that minimize the lsq_residual of drug
        """

        def lsq(parameters):
            return self.evaluate_lsq_residual(parameters, True)

        bounds = ((np.min(self.dose_data), np.max(self.dose_data)),
                  (1, 20),
                  (1e-6, 1))
        initial_values = self._get_optimizations_starts(bounds, n_starts)

        minimum_value = float('inf')
        minimum_parameters = None
        for i in range(n_starts):
            solution = minimize(lsq, initial_values[i], method='TNC', jac=True, bounds=bounds)
            if solution.fun < minimum_value:
                minimum_value = solution.fun
                minimum_parameters = solution.x
        self.parameters = minimum_parameters
        return minimum_parameters

    def _get_optimizations_starts(self,
                                  bounds: tuple,
                                  n_starts: int = 10):
        """Samples initial values in the bounds via Latin Hypercube sampling.

        Parameters
        ----------
        bounds: tuple
            the bounds in which sampled values should be

        n_starts: int
            number of sampled values

        Returns
        -------
        initialValues:
            initialValues for parameter optimization
        """
        initial_values = [[0] * 3 for i in range(n_starts)]
        perm_a = np.random.permutation(n_starts)
        perm_n = np.random.permutation(n_starts)
        perm_s = np.random.permutation(n_starts)
        for i in range(n_starts):
            initial_values[i] = [bounds[0][0] + (bounds[0][1] - bounds[0][0]) / n_starts * (perm_a[i] + 0.5),
                                 bounds[1][0] + (bounds[1][1] - bounds[1][0]) / n_starts * (perm_n[i] + 0.5),
                                 bounds[2][0] + (bounds[2][1] - bounds[2][0]) / n_starts * (perm_s[i] + 0.5)]
        return initial_values

    def get_sigma2(self):
        sum = 0
        a = self.parameters[0]
        n = self.parameters[1]
        s = self.parameters[2]
        x = self.dose_data
        y = self.response_data
        d = len(x)
        for i in range(d):
            sum += (s * x[i] ** n / (a ** n + x[i] ** n) - y[i]) ** 2

        return sum / d

    def _set_dose_and_response(self,
                               dose_data: np.array,
                               response_data: np.array):
        """
        sets dose and response data. Checks dimensions



        """

        if len(dose_data) == len(response_data):
            self.dose_data = dose_data
            self.response_data = response_data
        else:
            raise RuntimeError

    def _set_monotony(self):
        """
        sets monotone_increasing flag of drug
        """
        max = np.argmax(self.response_data)
        min = np.argmin(self.response_data)

        if self.response_data[min] < self.response_data[max]:
            self.monotone_increasing = True
        else:
            self.monotone_increasing = False

        return
