""" Dose Response Models via Hill Curves are defined here.
"""
import numpy as np
from .DoseResponseModelBase import DoseResponseModelBase


class HillModel(DoseResponseModelBase):

    def __init__(self,
                 dose_data: np.array = None,
                 response_data: np.array = None,
                 monotone_increasing: bool = True,
                 control_response: float = 0,
                 lb: np.array = None,
                 ub: np.array = None):
        """
        ToDo

        The hill curve is parametrized as

                        w_0 + s*x^n /(a+x^n)

        parameters:
            [a, n, s]

        """
        super().__init__(dose_data=dose_data,
                         response_data=response_data,
                         monotone_increasing=monotone_increasing,
                         control_response=control_response,
                         lb=lb,
                         ub=ub)

        self.parameters = np.nan * np.array([1, 1, 1])
        self._n_parameters = 3

    def get_response(self,
                     dose: np.array,
                     parameters: np.array = None,
                     gradient: bool = False):
        """
        TODO
        w_0 + s*x^n /(a+x^n)
        """
        if parameters is None:
            parameters = self.parameters

        a, n, s = parameters[0], parameters[1], parameters[2]

        unscaled_response = dose**n/(a**n + dose**n)

        if self.monotone_increasing:
            response = self.control_response + s * unscaled_response
        else:
            response = self.control_response - s * unscaled_response

        if not gradient:
            return response

        # gradient = True => gradient needs to be computed

        if dose == 0:
            return response, np.zeros_like(parameters)

        grad = unscaled_response * np.array([-s*n*a**(n-1)/(a**n + dose**n),
                                             s*a**n * np.log(dose/a)/(a**n + dose**n),
                                             1])

        # change sign for monotone decreasing drugs
        if not self.monotone_increasing:
            grad = -grad

        return response, grad

    def _get_default_parameter_bounds(self):
        """
        Returns the default parameter bounds. They are defined via
            - a: [min_dose, max_dose]
            - n: [0, 10]
            - s: [min_response, max(max_response, 1)]

        The bounds for s need to be rescaled w.r.t the control response and
        and consider, if the model is increasing or decreasing.

        :return:
            lb, ub: np.array, dimension (3, )
        """

        if self.monotone_increasing:
            s_min = np.min(self.response_data) - self.control_response
            s_max = np.max([np.max(self.response_data)-self.control_response, 1])
        else:
            s_min = self.control_response - np.max(self.response_data)
            s_max = np.max([self.control_response - np.min(self.response_data), 1])

        lb = np.array([np.min(self.dose_data),
                       1,
                       s_min])

        ub = np.array([np.max(self.dose_data),
                       10,
                       s_max])

        return lb, ub
