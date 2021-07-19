import abc
import numpy as np
import warnings
from typing import List

import pypesto
import pypesto.optimize as optimize
import pypesto.sample as sample


class DoseResponseModelBase(abc.ABC):
    """
    A DoseResponseModel stores a parametric representation of a dose response
    curve together with dose response data.

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
        response of the control experiment with dose zero

    lb: np.array
        lower bound of the model parameters.

    ub: np.array
        upper bound of the model parameters.

    """

    def __init__(self,
                 dose_data: np.array = None,
                 response_data: np.array = None,
                 monotone_increasing: bool = True,
                 control_response: float = 0,
                 lb: np.array = None,
                 ub: np.array = None,
                 parameter_names: List[str] = None):
        """
        Constructor
        """

        self.parameters = None
        self.n_parameters: int = None

        self.monotone_increasing = monotone_increasing
        self.control_response = control_response

        # set dose-response data:

        if dose_data is not None and response_data is not None:

            self._set_dose_and_response(dose_data, response_data)

        elif (dose_data is not None) is not (response_data is not None):

            warnings.warn("Only one of dose_data and response_data is "
                          "provided. Therefore this is neglected.")

            self.dose_data = None
            self.response_data = None

        else:

            self.dose_data = None
            self.response_data = None

        # set bounds
        if lb is not None:
            self.lb = lb
        else:
            self.lb = self._get_default_parameter_bounds()[0]

        if ub is not None:
            self.ub = ub
        else:
            self.ub = self._get_default_parameter_bounds()[1]

        self.pypesto_result: pypesto.result = None
        self.n_samples: int = 0

        self._set_fitting_problem(parameter_names=parameter_names)

    @property
    def has_samples(self):
        """Returns True, if the drug got sampled, False otherwise.
        (Samples are stored in self.pypesto_result.sample_result`)"""
        return self.n_samples > 0

    @abc.abstractmethod
    def get_response(self,
                     dose: float,
                     parameters: np.array = None,
                     gradient: bool = False):
        """
        Computes the response of the dose response model.
        If `gradient==True`, the gradient is returned as well.

        Parameters:
        -----------
        dose: float
            Drug Dose

        parameters: np.array
            Parameters. If none are provided, self.parameters is used.

        gradient: bool
            If `gradient==True`, the gradient is returned as second result.
        """
        raise NotImplementedError

    def get_multiple_responses(self,
                               doses: np.array,
                               parameters: np.array = None,
                               gradient: bool = False):
        """
        Similar to get_response, but for multiple doses


        Parameters:
        -----------
        doses: np.array
            Array of doses.

        parameters: np.array
            Parameters. If none are provided, self.parameters is used.

        gradient: bool
            If `gradient==True`, the gradient is returned as second result.
        """
        if gradient:

            responses = np.nan * np.ones_like(doses)
            gradients = np.nan * np.ones((self.n_parameters, doses.size))

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
                              parameters: np.array = None,
                              gradient: bool = False):
        r"""
        Evaluates the Least Squares Residual $\sum_{x_i} (f(x_i)-d_i)^2$.
        If `gradient==True`, the gradient is returned as second result.


        Parameters:
        -----------
        parameters: np.array
            Parameters. If none are provided, self.parameters is used.

        gradient: bool
            If `gradient==True`, the gradient is returned as second result.
        """
        if self.response_data is None:
            raise RuntimeError("Drug is missing response data. "
                               "Therefore the residual can not be computed.")

        if parameters is None:
            if self.parameters is None:
                raise ValueError("Drug must have parameters, if no parameters "
                                 "are provided to evaluate_lsq_residual.")
            parameters = self.parameters

        if gradient:
            responses, gradients = \
                self.get_multiple_responses(self.dose_data,
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
                       n_starts: int = 10,
                       optimizer: 'optimize.optimizer' = None,
                       parallelize_fitting: bool = False,
                       store_optimizer_output: bool = True):
        """
        Performs the Fitting of the dose-response curves (using pypesto).
        Sets the parameters of the optimization procedure to be the the
        DoseResponseModels parameters.

        Parameters:
        -----------
        n_starts:
            number of optimization starts

        optimizer:
            Optimizer, from the list of pypesto Optimizer.

        parallelize_fitting
            Indicates, whether fitting is parallelized.

        store_optimizer_output:
            Indicates if the result object (including metadata) is stored.
        :return:
        """
        if parallelize_fitting:
            engine = pypesto.engine.MultiProcessEngine()
        else:
            engine = None

        # fitting
        pypesto_result = optimize.minimize(
            self.pypesto_problem,
            optimizer=optimizer,
            startpoint_method=pypesto.startpoint.latin_hypercube,
            result=self.pypesto_result,
            engine=engine,
            n_starts=n_starts,
            progress_bar=False)

        # Extract best fit.
        self.parameters = pypesto_result.optimize_result.as_list()[0]['x']

        # store all meta information, if desired
        if store_optimizer_output:
            self.pypesto_result = pypesto_result

    def sample_parameters(self,
                          n_samples: int,
                          sampler: sample.Sampler = None):
        """
        Samples parameters according to the posterior given the
        dose response data. Stores the samples in
        `self.pypesto_result.sample_result`.

        Parameters:
        -----------
        n_samples:
            number of samples drawn.
        sampler:
            pypesto sampler, that shall be used.
        """
        if self.pypesto_result is None:
            self.fit_parameters()

        self.pypesto_result = sample.sample(self.pypesto_problem,
                                            n_samples=n_samples,
                                            sampler=sampler,
                                            result=self.pypesto_result)

        self.n_samples = n_samples

        # compute auto-correlation, burn-in and effective sample size.
        sample.effective_sample_size(self.pypesto_result)

        # check convergence.
        if self.pypesto_result.sample_result['effective_sample_size'] is None:
            raise RuntimeWarning('Sampling has not converged yet.')

    def get_ith_parameter_sample(self,
                                  i: int):
        """
        Returns the i-th parameter sample.
        """
        if i > self.n_samples:
            raise IndexError("Index exceeds number of samples.")
        else:
            return self.pypesto_result.sample_result['trace_x'][0][i]

    def _set_dose_and_response(self,
                               dose_data: np.array,
                               response_data: np.array):
        """
        Sets dose and response data. Checks dimensions

        Raises:
            ValueError: if doses and responses do not have equal size...
        """
        # no data provided
        if (dose_data is None) and (response_data is None):

            self.dose_data = None
            self.response_data = None

        # incomplete data is provided
        elif (dose_data is not None) is not (response_data is not None):

            warnings.warn("Only one of dose_data and response_data is "
                          "provided. Therefore this is neglected.")

            self.dose_data = None
            self.response_data = None

        else:
            # type cast, in case e.g. a list is provided...
            dose_data = np.array(dose_data)
            response_data = np.array(response_data)

            # miss-match in size
            if dose_data.size != response_data.size:
                raise ValueError('Dose and response data do '
                                 'not have equal size.')

            # assign values, as all sanity checks passed.
            self.dose_data = dose_data
            self.response_data = response_data

    def _set_fitting_problem(self,
                             parameter_names: List[str] = None):
        """
        Sets the pypesto problem, that is used for fitting and sampling
        of dose response models.

        Parameters:
        -----------
        parameter_names:
            List of parameter names.
        """
        def obj_fun(parameters: np.array):
            return self.evaluate_lsq_residual(parameters, gradient=True)

        objective = pypesto.Objective(fun=obj_fun,
                                      grad=True)

        if (self.lb is not None) and (self.ub is not None):
            self.lb, self.ub = self._get_default_parameter_bounds()

        self.pypesto_problem = pypesto.Problem(objective=objective,
                                               lb=self.lb,
                                               ub=self.ub,
                                               x_names=parameter_names)

    @abc.abstractmethod
    def _get_default_parameter_bounds(self):
        raise NotImplementedError

    @abc.abstractmethod
    def check_bliss_consistency(self):
        """
        Returns True, if the response of the dose response model is bound
        to [0, 1].

        In order to be valid for the Bliss combination model, every dose
        response model needs to be bounded to the interval [0, 1].
        (So that the response is interpretable as probability).

        Further test f(0)==0 for monotone increasing and f(0)==1 for monotone
        decreasing drugs.
        """
        raise NotImplementedError
