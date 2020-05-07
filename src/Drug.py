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

                        w_0 + s*x^n /(a+x^n)

    Attributes
    ----------

    parameters: np.array
        [a, n, s]
        w_0 der Effekt ohne Drug (bekannt)
        s: der maximale Effekt
        n: Hill-Coeffizient
        a: Quasi der Half-Max (Half-Max ist a^{1/n}
        # TODO: Docu parameters a, n, s, what do they mean?

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

    # Docu TODO
    """

    def __init__(self,
                 dose_data: np.array = None,
                 response_data: np.array = None,
                 monotone_increasing: bool = True,
                 control_response: float = 0):
        """
        Constructor
        """

        self.parameters = None

        self.monotone_increasing = monotone_increasing
        self.control_response = control_response

        if dose_data is not None and response_data is not None:
            self._set_dose_and_response(dose_data, response_data)

    def get_response(self,
                     dose: float,
                     parameters: np.array = None,
                     gradient: bool = False):
        """
               dose -> Dose
               parameters -> parameters, if none use the parameters from the Drug
               gradient -> if True, you also return the gradient...
               #drug class hat eigentlich noch w_o und monotone incresing...
               gives single dose response
               """
        if parameters is None:
            parameters = self.parameters
        a = parameters[0]
        n = parameters[1]
        s = parameters[2]
        response_value: float = s * dose ** n / (a + dose ** n)
        if not gradient:
            return response_value
        grad = np.array([np.nan, np.nan, np.nan])
        grad[0] = -s * dose ** n / ((a + dose ** n) ** 2)
        grad[1] = a * s * dose ** n * math.log(dose) / ((a + dose ** n) ** 2)
        grad[2] = dose ** n / (a + dose ** n)
        return response_value, grad


    def get_multiple_responses(self,
                               doses: np.array,
                               parameters: np.array = None,
                               gradient: bool = False):
        """
               dose -> Dose
               parameters -> parameters, if none use the parameters from the Drug
               gradient -> if True, you also return the gradient...
          """
        l = len(doses)
        responses = np.nan * np.ones((l, 1))
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
                              gradient: bool):
        """
        Evaluates the LSQ residual for given parameters AND returns gradient
        """
        lsq_residual = 0
        for i in range(len(self.response_data)):
            lsq_residual += (self.response_data[i] - (
                    self.control_response + parameters[2] * self.dose_data[i] ** parameters[1] /
                        (parameters[0] + self.dose_data[i] ** parameters[1]))) ** 2
        lsq_residual = lsq_residual
        if not gradient:
            return lsq_residual
        (responses, responses_grad) = self.get_multiple_responses(self.dose_data, parameters, True)
        grad = responses_grad.dot(np.add(-2*self.dose_data,2*responses))
        return lsq_residual, grad


    def fit_parameters(self,
                       n_starts: int = 10):
        """
        fits parameters to data and stores new parameters in drug
        """

        def lsq(parameters):
            return self.evaluate_lsq_residual(parameters, False)

        def lsq_grad(parameters):
            (lsq, grad) = self.evaluate_lsq_residual(parameters, True)
            return grad


        b = (0, 9)
        bounds = (b, b, b)
        initialValues = self._get_optimizations_starts(bounds, n_starts)

        minimum_value = float('inf')
        minimum_parameters = None
        for i in range(n_starts):
            solution = minimize(lsq, initialValues[i], method='SLSQP', bounds=bounds)#Todo gradienten mit einbeziehen
            if solution.fun < minimum_value:
                minimum_value = solution.fun
                minimum_parameters = solution.x
        self.parameters = minimum_parameters
        return minimum_parameters


    def _get_optimizations_starts(self,
                                  bounds: tuple,
                                  n_starts: int = 10):
        """
        Sample initial values in the bounds via Latin Hypercube sampling...
        """
        initialValues = [[0] * 3 for i in range(n_starts)]
        perm_a = np.random.permutation(n_starts)
        perm_n = np.random.permutation(n_starts)
        perm_s = np.random.permutation(n_starts)
        for i in range(n_starts):
            initialValues[i] = [bounds[0][0] + (bounds[0][1] - bounds[0][0])/ n_starts* (perm_a[i] + 0.5),
                                bounds[1][0] + (bounds[1][1] - bounds[1][0])/ n_starts* (perm_n[i] + 0.5),
                                bounds[2][0] + (bounds[2][1] - bounds[2][0])/ n_starts* (perm_s[i] + 0.5)]
        return initialValues


    def _set_dose_and_response(self,
                               dose_data: np.array,
                               response_data: np.array):
        """
        sets dose and response data. Checks dimensions


        """

        if dose_data.size == response_data.size:
            self.dose_data = dose_data
            self.response_data = response_data
        else:
            raise RuntimeError


if __name__ == '__main__':

    w_0 = 0.5
    dose_data = np.array([1, 2, 3, 4])
    response_data = np.array([5, 6, 7, 8])
    test_drug: Drug = Drug(dose_data, response_data, monotone_increasing=True,
                     control_response=w_0)
    test_drug.parameters = np.array([3, 2, 1])

    #test fit_parameters

    print(test_drug.parameters)

    solution = test_drug.fit_parameters(10)
    print(solution)
    print(test_drug.parameters)

