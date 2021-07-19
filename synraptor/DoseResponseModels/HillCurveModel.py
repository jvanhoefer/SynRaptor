""" Dose Response Models via Hill Curves are defined here.
"""
import numpy as np
import warnings
from typing import List

from .DoseResponseModelBase import DoseResponseModelBase


class HillCurveModel(DoseResponseModelBase):

    def __init__(self,
                 dose_data: np.array = None,
                 response_data: np.array = None,
                 monotone_increasing: bool = True,
                 control_response: float = 0,
                 lb: np.array = None,
                 ub: np.array = None,
                 parameter_names: List[str]=None):
        """
        ToDo

        The hill curve is parametrized as

                        w_0 + s*x^n /(a^n+x^n)

        parameters:
            [a, n, s]

        """
        if parameter_names is None:
            parameter_names = ['a', 'n_{Hill}', 's']

        super().__init__(dose_data=dose_data,
                         response_data=response_data,
                         monotone_increasing=monotone_increasing,
                         control_response=control_response,
                         lb=lb,
                         ub=ub,
                         parameter_names=parameter_names)

        self.n_parameters = 3

    def get_response(self,
                     dose: np.array,
                     parameters: np.array = None,
                     gradient: bool = False):
        """
        TODO
        w_0 + s*x^n /(a^n+x^n)
        """
        if parameters is None:
            parameters = self.parameters

        a, n, s = parameters[0], parameters[1], parameters[2]

        # compute unscaled response
        if dose == 0:
            unscaled_response=0
        else:
            unscaled_response = dose**n/(a**n + dose**n)

        # scale response
        if self.monotone_increasing:
            response = self.control_response + s * unscaled_response
        else:
            response = self.control_response - s * unscaled_response

        if not gradient:
            return response

        else:  # gradient = True => compute gradient

            if dose == 0:
                return response, np.zeros_like(parameters)

            # deal with with a == 0 in log(dose/a)
            if a == 0:
                log_d_a = - np.inf
            else:
                log_d_a = np.log(dose/a)


            grad = unscaled_response * np.array(
                [-s*n*a**(n-1)/(a**n + dose**n),
                 s*a**n * log_d_a/(a**n + dose**n),
                 1])

            # change sign for monotone decreasing drugs
            if not self.monotone_increasing:
                grad = -grad

            return response, grad

    def _get_default_parameter_bounds(self):
        """
        Returns the default parameter bounds. They are defined via
            - a: [min(min_dose, 1e-4), max_dose]
            - n: [1, 10]
            - s: [min_response, max(max_response, 1)]

        The bounds for s need to be rescaled w.r.t the control response and
        and consider, if the model is increasing or decreasing.

        :return:
            lb, ub: np.array, dimension (3, )
        """

        if (self.dose_data is None) or (self.response_data is None):
            warnings.warn("No dose-response data available. "
                          "Hence the lower/upper bounds are chosen "
                          "generically and might be unphysiological for the "
                          "given dose response curve.")
            return np.array([1e-4, 1, 0]), np.array([1e4, 10, 1])

        if self.monotone_increasing:
            s_min = np.min(self.response_data) - self.control_response
            s_max = np.max([np.max(self.response_data)
                            - self.control_response, 1])
        else:
            # ensure that s_min is non-negative
            s_min = max(self.control_response - np.max(self.response_data), 0)
            # ensure, that s_max <= 1
            s_max = np.max([self.control_response
                            - np.min(self.response_data), 1])

        lb = np.array([min(np.min(self.dose_data), 1e-4),
                       1,
                       s_min])

        ub = np.array([np.max(self.dose_data),
                       10,
                       s_max])

        return lb, ub

    def check_bliss_consistency(self):
        """
        Checks if the response is bound to [0, 1].
        """
        if self.parameters is None:
            raise RuntimeError("The Hill Curve does not contain parameters. "
                               "Therefore the bliss consistency can not be "
                               "checked.")
        if self.monotone_increasing:
            # check if w_0==0 and  the max response w_0 + s \in [0, 1]
            return (self.control_response == 0) and \
                   (0 <= self.control_response + self.parameters[2] <= 1)
        else:  # monotone decreasing
            # check if w_0==0 and the max response w_0 - s \in [0, 1]
            return (self.control_response == 1) and \
                   (0 <= self.control_response - self.parameters[2] <= 1)
