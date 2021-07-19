import abc
import numpy as np

import pypesto
import pypesto.optimize as optimize
import pypesto.profile as profile
import pypesto.sample as sample

from scipy.optimize import bisect

from scipy.stats import norm

from typing import List, Union
from ..DoseResponseModels import DoseResponseModelBase


# ToDo parameters as input: make this consistent between Union[np.array, List[List]] and np.ndarray


class CombinationModelBase(abc.ABC):
    """
    A CombinationModel stores a a list of DoseResponseModels and provides
    functionality to compute predictions for dose combinations and the
    significance of a combination experiment.

    Attributes
    ----------

    drug_list:
        List of combined drugs.
    n_drugs:
        Number of drugs, that are combined.
    sigma:
        variance of the (gaussian) noise of the measurements.
        None, if not all drugs have dose-response data.
    """

    def __init__(self,
                 drug_list: List[DoseResponseModelBase]):
        """
        Constructor.
        """
        self.drug_list = drug_list
        self.n_drugs = len(drug_list)

        self._check_drug_consistency()

        if self._check_all_drugs_have_data():
            self._set_sigma()
            self._set_combined_optimize_result()

        else:
            self.sigma = None
            self.single_drug_result = None

    @abc.abstractmethod
    def get_combined_effect(self,
                            dose_combination: List,
                            parameters: Union[np.array, List[List]] = None,
                            gradient: bool = False):
        """
        Returns the effect of a dose combination predicted by the
        combination model.

        Parameters:
        -----------

        dose_combination:
            List of doses of the individual drugs.
        parameters:
            parameters of the individual drugs. Should be the concatenation
            of the different individual drug parameters.
        gradients:
            If True, the gradient w.r.t the parameters is returned as
            second output.
        """
        raise NotImplementedError

    def get_index_range(self,
                        idx: int):
        """
        parameters[i_min:i_max] correspond to drug number idx.

        Parameters:
        -----------
        idx:
            Index of the corresponding drug.
        """
        parameter_idx_start = 0

        for i in range(idx):
            parameter_idx_start += self.drug_list[i].n_parameters

        return parameter_idx_start, \
            parameter_idx_start+self.drug_list[idx].n_parameters

    def get_single_drug_response(self,
                                 idx: int,
                                 dose: float,
                                 parameters: Union[np.array, List[List]] = None,
                                 gradient: bool = False):
        """
        Returns the response of drug with index idx in the drug list.

        Parameters:
        -----------
        idx:
            response of the drug in self.drug_list[idx] is returned
        dose:
            dose of the drug
        parameters:
            parameters, if None, self.drug_list[idx].parameters are used.
        gradient:
            If True, gradient is returned as second output.
        """
        # if no parameters are given, use deafult
        if parameters is None:
            parameter = None

        # parameters are a list
        if isinstance(parameters, list):

            # parameters is a long list of entries
            if isinstance(parameters[0], float):
                i_min, i_max = self.get_index_range(idx)
                parameter = parameters[i_min:i_max]
            else: # parameters is a list of lists
                parameter = parameters[idx]

        elif isinstance(parameters, np.ndarray):
            if parameters.ndim == 1:  # parameters is a long list of entries
                i_min, i_max = self.get_index_range(idx)
                parameter = parameters[i_min:i_max]
            else:
                parameter = parameters[idx]
        else:
            TypeError("parameters must be of the type list or np.ndarray.")

        return self.drug_list[idx].get_response(dose, parameter, gradient)

    def _get_single_drug_data_nllh(self,
                                   parameters: np.array = None,
                                   gradient: bool = False):
        """
        Computes the (negative log-)likelihood of the data of the single drugs.

        Parameters:
        -----------

        parameters:
            parameters of the individual drugs. Should be the concatenation
            of the different individual drug parameters.
        gradient: bool
            If True, the gradient w.r.t the parameters is returned as
            second output.
        """
        if self.sigma is None:
            raise RuntimeError(
                "Combination does not have a sigma value. "
                "This is typically the case, if not all individual drugs of a "
                "combination have dose-response data.")

        nllh = 0
        start_idx = 0

        if not gradient:

            for drug in self.drug_list:
                nllh += drug.evaluate_lsq_residual(
                    parameters[start_idx:start_idx+drug.n_parameters], gradient)

                start_idx += drug.n_parameters

            # rescale from lsq to nnlh
            return 1/(2*self.sigma) * nllh

        else:  # return gradient

            grad = np.zeros_like(parameters)

            for drug in self.drug_list:

                n, g = drug.evaluate_lsq_residual(
                    parameters[start_idx:start_idx+drug.n_parameters],
                    gradient)
                nllh += n
                grad[start_idx:start_idx+drug.n_parameters] = \
                    grad[start_idx:start_idx+drug.n_parameters] + g

                start_idx += drug.n_parameters

            # rescale by / 2*sigma to get the nllh instead of the lsq
            return 1/(2*self.sigma) * nllh, 1/(2*self.sigma) * grad

    def _get_full_data_nllh(self,
                            dose_combination: np.array,
                            response_data: Union[float, List],
                            parameters: np.array = None,
                            gradient: bool = False):
        """
        Computes the (negative log)likelihood of the data full data set.

        Parameters:
        -----------

        dose_combination:
            List of doses of the individual drugs.
            len(dose_combination) = self.n_drugs
        response_data:
            Response data. If a list is given, the responses are
            interpreted as replicates for the same dose combination.
        parameters:
            parameters of the individual drugs. Should be the concatenation
            of the different individual drug parameters.
        gradient: bool
            If True, the gradient w.r.t the parameters is returned as
            second output.
        """
        if isinstance(response_data, float):
            response_data = np.array([response_data])

        if not gradient:

            # contribution of the single drug data
            nllh = self._get_single_drug_data_nllh(parameters, gradient)

            # combined response and add contribution of combination data.
            predicted_response = self.get_combined_effect(dose_combination,
                                                          parameters,
                                                          gradient)

            for response in response_data:
                nllh += 1/(2*self.sigma) * (predicted_response-response)**2

            return nllh

        else:  # return gradient
            # contribution of the single drug data
            nllh, grad = self._get_single_drug_data_nllh(parameters,
                                                         gradient)

            # combined response/gradient
            predicted_response, predicted_response_gradient \
                = self.get_combined_effect(dose_combination,
                                           parameters,
                                           gradient)

            for response in response_data:
                nllh += 1/(2*self.sigma) * (predicted_response-response)**2
                grad += ((predicted_response-response)/self.sigma) \
                    * predicted_response_gradient

            return nllh, grad

    def get_likelihood_ratio_significance(self,
                                          dose_combination: np.array,
                                          response_data: Union[float, List],
                                          optimizer: optimize.Optimizer = None):
        """
        Computes the significance of a combination experiment based on
        validation intervals as introduced by Kreutz et al. 2012.

        Parameters:
        -----------
        dose_combination:
            List of doses of the individual drugs.
            len(dose_combination) = self.n_drugs
        response_data:
            Response data. If a list is given, the responses are
            interpreted as replicates for the same dose combination.
        optimizer:
            pyPESTO optimizer, that is used for the internal optimization.
        """

        def nllh_full_data(x):
            return self._get_full_data_nllh(dose_combination,
                                            response_data,
                                            parameters=x,
                                            gradient=True)

        # generate the pypesto Problem including the validation experiment
        nllh_full_data_objective = pypesto.Objective(
            fun=nllh_full_data, grad=True)

        lb, ub = self._get_combined_bounds()

        nllh_full_data_problem = pypesto.Problem(
            objective=nllh_full_data_objective, lb=lb, ub=ub)

        # compute the validation significance in pypesto
        return profile.validation_profile_significance(
            problem_full_data=nllh_full_data_problem,
            result_training_data=self.single_drug_result,
            optimizer=optimizer)

    def get_likelihood_ratio_ci(self,
                                dose_combination: List,
                                alpha: float = 0.95,
                                optimizer: optimize.Optimizer = None,
                                a_tol: float = 1e-3):
        """
        Determines the 100 * (1-alpha) percent confidence interval (of a
        validation experiment) using likelihood ratio based prediction
        intervals (via bisection).

        Parameters:
        dose_combination:
            Dose Combination, at which the prediction interval shall be
            evaluated.
        alpha:
            confidence level of prediction interval (in [0, 1.0])
        optimizer:
            pyPESTO optimizer, that is used for the internal optimization.
        a_tol:
            absolute tolerance, for terminating the bisection procedure.
        """
        if not (0 < alpha < 1):
            raise ValueError("For the significance must hold 0<alpha<1.")

        # this step would lead to a correct CI if there were no parameter
        # uncertainties involved.
        delta_step = norm.ppf(alpha + 1/2 * (1-alpha))

        # Get a starting value for finding lower and upper bounds
        c_min = self.get_combined_effect(dose_combination)
        c_max = c_min

        def lr_sig_plus_alpha_minus_1(response: float):
            """
            The boundaries of the CI have
            likelihood_ratio_significance(response) == 1 - alpha.


            Hence we look for the zeros of the function
            likelihood_ratio_significance(response) + alpha - 1"""
            lr_sig = self.get_likelihood_ratio_significance(
                dose_combination,
                response,
                optimizer)

            return lr_sig + alpha - 1

        n_iter = 0
        max_iter = 500

        # c_min:

        # iterate, until lstarting interval for bisection is found.
        while (lr_sig_plus_alpha_minus_1(c_min) > 0) and n_iter < max_iter:
            c_min -= delta_step
            n_iter += 1

        # if lower bound was not reached yet, return -inf,
        # else perform bisection.
        if n_iter == max_iter:
            c_min = -np.inf
        else:
            c_min = bisect(lr_sig_plus_alpha_minus_1,
                           c_min,
                           c_min+delta_step,
                           xtol=a_tol)

        # same for c_max:
        n_iter = 0

        while (lr_sig_plus_alpha_minus_1(c_max) > 0) and n_iter < max_iter:
            c_max += delta_step
            n_iter += 1

        if n_iter == max_iter:
            c_max = np.inf
        else:
            c_max = bisect(lr_sig_plus_alpha_minus_1,
                           c_max-delta_step,
                           c_max,
                           xtol=a_tol)

        return [c_min, c_max]

    def get_sampling_significance(self,
                                  dose_combination: np.array,
                                  response_data: Union[float, List],
                                  n_samples: int = 10000,
                                  sampler: sample.Sampler = None):
        """
        Tests the significance of mean of the corresponding dose-response data.
        I.e. computes predictions from the parameter samples (with an adapted
        variance) and compares the samples to the mean of the dose response data.
        Here the burn in of the samples is not used.

        The (1-alpha)/2 percentile of the samples is larger than the observed
        average response (for two-sided).

        (When comparing results: Be aware, that all tests in SynRaptor are
        two-sided tests.)

        Parameters:
        -----------
        dose_combination:
            List of doses of the individual drugs.
            len(dose_combination) = self.n_drugs
        response_data:
            Response data. If a list is given, the responses are
            interpreted as replicates for the same dose combination.
        n_samples
            number of samples, that should be taken
            (after throwing out the burn in.)
        sampler:
            Pypesto sampler, that is used for resampling non-converged drugs.
        """
        # compute mean response and adapt the variance
        try:
            n_replicates = len(response_data)
        except:
            n_replicates = 1

        sigma_data = self.sigma
        mean_response = np.mean(response_data)

        # compute predictions.
        predictions = self.get_sampling_predictions(dose_combination,
                                                    sigma=sigma_data,
                                                    n_replicates=n_replicates,
                                                    n_samples=n_samples,
                                                    sampler=sampler)
        # compute the percentiles.
        rank = np.sum(predictions < mean_response)/n_samples
        return 2 * np.abs(1/2 - rank)

    def get_sampling_ci(self,
                        dose_combination: List,
                        alpha: float = 0.95,
                        sigma: float = None,
                        n_replicates: int = 1,
                        n_samples: int = 10000,
                        sampler: sample.Sampler = None):
        """
        Computes sampling based confidence intervals of one
        validation experiment.

        I.e. 100 * (1 - alpha) percent of the sampled predictions are within
        the CI.

        Parameters:
        -----------
        dose_combination:
            List of doses of the individual drugs.
            len(dose_combination) = self.n_drugs
        alpha:
            confidence level of prediction interval (in [0, 1.0])
        sigma:
            Standard deviation of the measurement error.
            Default is the sigma of the combination model `self.sigma`.
        n_replicates:
            number of replicates. Sigma is scaled such that the CI for the
            sample mean over n_replicates replicates is returned.
        n_samples
            number of samples, that should be taken
            (after throwing out the burn in.)
        sampler:
            Pypesto sampler, that is used for resampling non-converged drugs.
        """
        if not (0 < alpha < 1):
            raise ValueError("For the significance must hold 0<alpha<1.")

        if sigma is None:
            sigma = self.sigma

        # compute and sort predictions.
        predictions = self.get_sampling_predictions(dose_combination,
                                                    sigma=sigma,
                                                    n_replicates=n_replicates,
                                                    n_samples=n_samples,
                                                    sampler=sampler)
        predictions = np.sort(predictions)

        # get min/max indices
        idx_min = int(alpha/2*n_samples)
        idx_max = n_samples - idx_min

        return [predictions[idx_min], predictions[idx_max]]

    def get_sampling_predictions(self,
                                 dose_combination: np.array,
                                 sigma: float = None,
                                 n_replicates: int = 1,
                                 n_samples: int = 10000,
                                 sampler: sample.Sampler = None):
        """
        Samples n_sample many predictions for standard deviation sigma

        Parameters:
        -----------
        dose_combination:
            List of doses of the individual drugs.
            len(dose_combination) = self.n_drugs
        sigma:
            Standard deviation of the measurement error.
            Default is the sigma of the combination model `self.sigma`.
        n_samples
            number of samples, that should be taken
            (after throwing out the burn in.)
        sampler:
            Pypesto sampler, that is used for resampling non-converged drugs.
        """

        if sigma is None:
            sigma = self.sigma

        sigma = sigma/np.sqrt(n_replicates)

        # check, if the minimal number of samples after burn in are reached
        # and re-sample otherwise...
        self._check_all_drugs_have_converged_samples(
            min_n_samples=n_samples,
            sampler=sampler)

        burn_ins = [drug.pypesto_result.sample_result['burn_in']
                    for drug in self.drug_list]

        # loop over the parameters and  compute predictions.
        predictions = np.nan * np.zeros(n_samples)

        for i in range(n_samples):
            parameters = [drug.get_ith_parameter_sample(burn_ins[idx]+i)
                          for idx, drug in enumerate(self.drug_list)]

            predictions[i] = self.get_combined_effect(dose_combination,
                                                      parameters)

        # add noise to predictions
        predictions = predictions + np.random.normal(scale=sigma,
                                                     size=n_samples)

        return predictions

    def _check_all_drugs_have_converged_samples(
            self,
            min_n_samples: int,
            sampler: sample.Sampler = None):
        """
        Checks, that all drugs have min_n_samples converged samples.
        Re-samples otherwise.
        """
        def check_drug(drug):
            """
            Checks if the drug needs to be resampled
            and estimates the number of required samples.
            """
            # drug is not sampled yet
            if drug.n_samples < min_n_samples:
                resample = True
                n_samples = int(1.2 * min_n_samples)

            # sampling did not converge yet.
            elif drug.pypesto_result.sample_result['burn_in'] is None:
                resample = True
                n_samples = int(1.2 * (drug.n_samples + min_n_samples))

            # not enough converged samples yet.
            elif drug.n_samples - \
                    drug.pypesto_result.sample_result['burn_in'] \
                    < min_n_samples:

                resample = True
                n_samples = drug.pypesto_result.sample_result['burn_in'] \
                    + int(1.2 * min_n_samples)

            # sampling converged with enough samples.
            else:
                resample = False
                n_samples = None

            return resample, n_samples

        for drug in self.drug_list:
            finished, n_samples_drug = check_drug(drug)

            while finished:
                # resample
                drug.sample_parameters(n_samples_drug,
                                       sampler=sampler)

                finished, n_samples_drug = check_drug(drug)

    def _check_all_drugs_have_data(self):
        """
        Checks if all drugs have dose response data.
        """
        for drug in self.drug_list:
            if (drug.dose_data is None) and (drug.response_data is None):
                return False

        return True

    def _check_drug_consistency(self):
        """
        Checks if all drugs of a combination are consistent, i.e. all share
        the same control response and all are either monotone increasing or
        decreasing.
        """
        control_response = self.drug_list[0].control_response
        increasing = self.drug_list[0].monotone_increasing

        for drug in self.drug_list:

            if control_response != drug.control_response:
                raise ValueError("All drugs of a combination must share their "
                                 "value of the control response.")

            if increasing is not drug.monotone_increasing:
                raise ValueError("All drugs of a combination must be either "
                                 "monotone increasing or decreasing.")

    def _set_sigma(self):
        """
        Sets the sample standard deviation (without bessel correction).
        """

        if not self._check_all_drugs_have_data():
            raise ValueError("All drugs must have dose-response data in order "
                             "to compute sigma.")

        n_data_total = 0
        total_squared_residual = 0

        for drug in self.drug_list:
            if drug.parameters is None:
                drug.fit_parameters()

            n_data_total += len(drug.response_data)
            total_squared_residual += drug.evaluate_lsq_residual()

        self.sigma = total_squared_residual/n_data_total

    def _set_combined_optimize_result(self):
        """
        Sets an pypesto.OptimizeResult object for the (combined) fitting
        problem.

        This is necessary, since validation intervals and sampling need a
        "combined" result object of the model consisting of all drugs and all
        data combined, while the fitting is done for the single drugs
        individually (for speed-up and numerical reasons...)
        """
        if self.sigma is None:
            self._set_sigma()

        # extract optimal parameters and compute nllh + gradient
        x_opt = np.concatenate(
            [drug.parameters for drug in self.drug_list])

        fval, grad = self._get_single_drug_data_nllh(parameters=x_opt,
                                                     gradient=True)

        # store everything in a dummy OptimizerResult
        dummy_optimizer_result = optimize.result.OptimizerResult(id='1',
                                                                 x=x_opt,
                                                                 fval=fval,
                                                                 grad=grad)

        # store the optimization result in a pypesto.Result
        result = pypesto.Result()
        result.optimize_result.append(dummy_optimizer_result)

        self.single_drug_result = result

    def _get_combined_bounds(self):
        """
        Extracts the lower and upper bounds from the single drug
        pypesto problems.
        """
        lb = np.concatenate([drug.pypesto_problem.lb for drug in self.drug_list])
        ub = np.concatenate([drug.pypesto_problem.ub for drug in self.drug_list])

        return lb, ub
