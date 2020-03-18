""" Single Drugs are defined here.
"""
import numpy as np
import scipy as sp


class Drug:
    """
    Drug stores a parametric representation of a dose response curve
    (=Hill curve) together with dose response data.

    The hill curve is parametrized as

                        w_0 + x^n /(a+x^n)

    Attributes
    ----------

    parameters: np.array
        [a, n, s]
        # TODO parameters a, n, s, what do they mean?

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

        self._set_dose_and_response(dose_data, response_data)

    def get_response(self,
                     dose: float,
                     parameters: np.array = None,
                     gradient: bool = False):
        """
        dose -> Dose
        parameters -> parameters, if none use the parameters from the Drug
        gradient -> if True, you also return the gradient...
        """
        # gives single dose response,
        # #(Only if theIF THEY ARE NOT None!!!!)
        # TODO: Docu, implement
        raise NotImplementedError

    def get_multiple_responses(self,
                               doses: np.array,
                               parameters: np.array = None,
                               gradient: bool = False):
        """
        dose -> Dose
        parameters -> parameters, if none use the parameters from the Drug
        gradient -> if True, you also return the gradient...
        """
        # similar to get_response, but for a dose vector
        # VECTORISE your implementation!!!!
        # TODO: Docu, implement
        raise NotImplementedError

    def evaluate_lsq_residual(self, parameters: np.array):
        # Evaluates the LSQ residual for given parameters AND returns gradient
        # VECTORIZE!!!

        raise NotImplementedError

    def fit_parameters(self, ):
        # Fit the data (LSQ here...), here you can use scipy optimizers...
        # try to get this as fast as possible...

        raise NotImplementedError

    def _get_initial_parameters(self,
                                bounds: np.array,
                                n_starts: int = 10):
        # Sample initial values in the bounds via Latin Hypercube sampling...

        raise NotImplementedError

    def _set_dose_and_response(self,
                               dose_data: np.array,
                               response_data: np.array):
        """
        sets dose and response data. Checks dimensions

        Raises:
            RuntimeError
        """

        if dose_data.size == response_data.size:

            self.dose_data = dose_data
            self.response_data = response_data
