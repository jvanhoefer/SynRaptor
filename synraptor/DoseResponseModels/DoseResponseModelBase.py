import abc
import numpy as np
import pypesto
import pypesto.optimize as optimize


class DoseResponseModelBase(abc.ABC):
    """
    A DoseResponseModel stores a parametric representation of a dose response curve
    together with dose response data.

    Attributes
    ----------

    parameters: np.array
        parameters of the model

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

    DoseResponseModel: DoseResponseModel
        Constructor

    # Docu TODO
    """

    def __init__(self,
                 dose_data: np.array = None,
                 response_data: np.array = None,
                 monotone_increasing: bool = True,
                 control_response: float = 0,
                 lb: np.array = None,
                 ub: np.array = None):
        """
        Constructor
        """

        self.parameters = None
        self._n_parameters: int = None

        self.monotone_increasing = monotone_increasing
        self.control_response = control_response

        self.lb = lb
        self.ub = ub
        self.pypesto_result = None

        if dose_data is not None and response_data is not None:
            self._set_dose_and_response(dose_data, response_data)
            self._set_fitting_problem()
        else:
            self.dose_data = None
            self.response_data = None

    @abc.abstractmethod
    def get_response(self,
                     dose: float,
                     parameters: np.array = None,
                     gradient: bool = False):
        """
        dose -> Dose
        parameters -> parameters, if none use the parameters from the DoseResponseModel
        gradient -> if True, you also return the gradient...
        """
        # gives single dose response,
        #
        # TODO: Docu, implement
        raise NotImplementedError

    def get_multiple_responses(self,
                               doses: np.array,
                               parameters: np.array = None,
                               gradient: bool = False):
        """
        dose -> Dose
        parameters -> parameters, if none use the parameters from the DoseResponseModel
        gradient -> if True, you also return the gradient...
        """
        # TODO: Docu
        if gradient:

            responses = np.nan * np.ones_like(doses)
            gradients = np.nan * np.ones((self._n_parameters, doses.size))

            for i in range(doses.size):
                responses[i], gradients[:, i] \
                    = self.get_response(doses[i],
                                        parameters=parameters,
                                        gradient=True)
            return responses, gradients

        else:  # gradient = False
            return np.array([self.get_response(d, parameters=parameters)
                             for d in doses])

    def evaluate_lsq_residual(self,
                              parameters: np.array,
                              gradient: bool):
        # TODO Docu
        if not self.response_data:
            raise RuntimeError("Drug is missing response data. Therefore the LSQ residual can not be computed.")

        if gradient:
            responses, gradients = self.get_multiple_responses(self.dose_data,
                                                               parameters=parameters,
                                                               gradient=True)
            residuals = responses - self.response_data

            return np.sum(residuals**2), 2 * np.dot(gradients, residuals)

        else:  # gradient = False
            residuals = (self.get_multiple_responses(self.dose_data,
                                                     parameters=parameters)
                         - self.response_data)
            return np.sum(residuals**2)

    def fit_parameters(self,
                       optimizer: 'optimize.optimizer' = None,
                       n_starts: int = 10,
                       paral_fitting: bool = False,
                       store_optimizer_output: bool = False):
        # Fit the data (LSQ here...), here you can use scipy optimizers...
        # Store the parameters as self.parameters afterwards
        # TRY TO GET THIS AS FAST AS POSSIBLE, WILL BE CALLED THOUSANDS OF TIMES!!!

        self._set_fitting_problem()

        # initialize for parallelization
        if paral_fitting:
            engine = pypesto.engine.MultiProcessEngine()
        else:
            engine = None

        # fitting
        pypesto_result = optimize.minimize(self.pypesto_problem,
                                           optimizer=optimizer,
                                           engine=engine,
                                           n_starts=n_starts)

        # Extract best fit.
        self.parameters = pypesto_result.optimize_result.as_list()[0]['x']

        # store all meta information, if desired
        if store_optimizer_output:
            self.pypesto_result = pypesto_result

    def _get_optimizations_starts(self,
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
            ValueError: if doses and responses do not have equal size...
        """

        if dose_data.size == response_data.size:
            self.dose_data = dose_data
            self.response_data = response_data
        else:
            raise ValueError('Dose and response data do not have the '
                             'equal size.')

    def _set_fitting_problem(self):
        """
        sets the pypesto problem, that can be used for fitting and sampling of dose response models.
        """
        objective = pypesto.Objective(fun=self.evaluate_lsq_residual,
                                      grad=True)

        if (self.lb is not None) and (self.ub is not None):
            self.lb, self.ub = self._get_default_parameter_bounds()

        self.pypesto_problem = pypesto.Problem(objective=objective,
                                               lb=self.lb,
                                               ub=self.ub)

    @abc.abstractmethod
    def _get_default_parameter_bounds(self):
        raise NotImplementedError
