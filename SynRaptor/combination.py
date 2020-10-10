"""Combinations of multiple drugs are defined here.
"""
import math
import numpy as np
import numpy.matlib
import scipy as sp
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import chi2


class Combination:
    """
    Combination stores a list of drugs and gives all functionality
    to evaluate synergy null models and derive synergy.

    Attributes
    ----------
    drug_list: list
        list of drugs

    Methods
    -------
    get_loewe_response:
        calculates the loewe prediction

    get_bliss_response:
        calculates the bliss prediction

    _check_bliss_requirements:
        checks if bliss requirements are met

    get_hand_response:
        calculates the hand prediction

    get_hsa_response:
        calculates the hsa prediction

    evaluate_rss_single_drug_data:
        evaluates -2 * log likelihood of data without validation data

    evaluate_validation_residual:
        calculates the validation residual

    fit_to_full_data:
        fits drug parameters to data including validation experiment

    _matrix_to_vector:
            reshapes parameters matrix to vector

    _vector_to_matrix:
        reshapes parameters vector to matrix

    _drug_list_to_parameters:
        creates array containing the parameters of self.drug_list

    combination_response:
        returns the null model response function belonging to the given null model

    old_fit_sum_likelihood:
        calculates the likelihood using the sum statistic regarding the drug parameters fitted from single drug data

    new_fit_sum_likelihood:
        optimizes the sum likelihood

    volume_significance:
        computes the significance level for a given dose and measurement and null model using the sum statistic

    get_significance:
        compute the significance level for a given dose and measurements and null_model

    _check_drug_consistency:
        checks if all drugs in drug_list have the same characteristics

    _set_sigma2:
        compute the (optimal) sigma for the given drugs in drug_list using hierarchical optimization

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

        if not np.any(dose_combination):

            if self.drug_list[0].monotone_increasing:
                effect = 0
            else:
                effect = 1

        else:

            def loewe_equation(effect):
                if effect == 0:
                    RuntimeWarning('evaluate loewe effect = 0')

                s = sum(dose_combination[i] / self.drug_list[i].inverse_evaluate(effect, parameters[i]) for i in
                        range(number_of_drugs))
                return s - 1

            if self.drug_list[0].monotone_increasing:
                a = self.drug_list[0].control_response + 1e-4
                b = self.drug_list[0].control_response + np.min([drug.parameters[2] for drug in self.drug_list]) - 1e-4

            else:
                a = self.drug_list[0].control_response - 1e-4
                b = self.drug_list[0].control_response - np.max([drug.parameters[2] for drug in self.drug_list]) + 1e-4

            effect = sp.optimize.bisect(loewe_equation, a, b)

            for i in range(len(self.drug_list)):

                if (self.drug_list[i].parameters[2] - self.drug_list[i].control_response + effect) < 0:

                    if gradient:
                        return self.get_hsa_response(dose_combination, True, parameters)
                    else:
                        return self.get_hsa_response(dose_combination, False, parameters)

        if not gradient:
            return effect

        def get_parameters(i: int):
            a, n, s = parameters[i][0], parameters[i][1], parameters[i][2]
            return a, n, s

        def set_d(effect):
            """
            Calculates d = y - w_0 for monotone increasing and d = w_0 - y for monotone decreasing Hill curves.
            """
            if self.drug_list[0].monotone_increasing:
                d = effect - self.drug_list[0].control_response
            else:
                d = self.drug_list[0].control_response - effect
            return d

        def di_dy(y,
                  i: int):  # index of drug in drug_list
            """
            Calculates the derivative of 1/f_i^{-1}(y)
            """
            a, n, s = get_parameters(i)
            d = set_d(y)

            if (s - d) < 0:
                return 0

            if self.drug_list[0].monotone_increasing:
                return - s * a ** n * (a ** n * d / (s - d)) ** (1 / (1 - n)) / (n * (s - d) ** 2)
            else:
                return s * a ** n * (a ** n * d / (s - d)) ** (1 / (1 - n)) / (n * (s - d) ** 2)

        df_dy = sum(dose_combination[i] * di_dy(effect, i) for i in range(number_of_drugs))  # calculates dF/dy

        def dy_da_i(y,
                    i: int):
            """
            Calculates dy/da_i
            """
            a, n, s = get_parameters(i)
            d = set_d(y)

            if (s - d) < 0:
                return 0

            return dose_combination[i] * (a ** n * d / (s - d)) ** (-1 / n) / a / df_dy

        def dy_dn_i(y,
                    i: int):
            """
            Calculates dy/dn_i
            """
            a, n, s = get_parameters(i)
            d = set_d(y)

            if d / (s - d) < 0.00001:

                if d / (s - d) > -0.00001:
                    raise RuntimeError('can not calculate log(0)')
                return 0

            if df_dy < 0.0001:

                if df_dy > -0.0001:
                    raise RuntimeError('can not divide by zero')
                return 0

            return - dose_combination[i] * (d / (s - d)) ** (-1 / n) * math.log(d / (s - d)) / (
                    a * n ** 2) / df_dy  # a,n,w,s,y are positive

        def dy_ds_i(y,
                    i: int):
            """
            Calculates dy/ds_i
            """
            a, n, s = get_parameters(i)
            d = set_d(y)
            return - dose_combination[i] * (a ** n * d / (s - d)) ** (-1 / n) / (n * (s - d)) / df_dy

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

        number_of_drugs = len(dose_combination)

        if parameters is None:
            parameters = [drug.parameters for drug in self.drug_list]

        self._check_bliss_requirements(parameters)

        # For monotone increasing drugs the Bliss response is 1-prod(1-y_i)
        if self.drug_list[0].monotone_increasing:

            if not gradient:
                return 1 - np.prod(
                    [1 - self.drug_list[i].get_response(dose_combination[i], parameters[i])
                     for i in range(number_of_drugs)])

            response_grad_matrix = [self.drug_list[i].get_response(dose_combination[i], parameters[i], True) for i in
                                    range(number_of_drugs)]
            one_minus_responses = [1 - response_grad_matrix[i][0] for i in range(number_of_drugs)]
            prod = np.prod(one_minus_responses)

            grad = - [prod / (one_minus_responses[i]) * response_grad_matrix[i][1] for i in range(number_of_drugs)]
            grad = self._matrix_to_vector(grad)  # now gradient looks like [a0 n0 s0 a1 n1 s1 ...]
            return 1 - prod, grad
        # For monotone decreasing drugs the Bliss response is prod(y_i)
        else:
            if not gradient:
                return np.prod([self.drug_list[i].get_response(dose_combination[i], parameters[i]) for i in
                                range(number_of_drugs)])

            response_grad_matrix = [self.drug_list[i].get_response(dose_combination[i], parameters[i], True) for i in
                                    range(number_of_drugs)]
            responses = [response_grad_matrix[i][0] for i in range(number_of_drugs)]
            prod = np.prod(responses)
            grad = [prod / responses[i] * response_grad_matrix[i][1] for i in range(number_of_drugs)]
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

        Parameters
        ----------
        parameters: np.array
            parameters of the hill curves
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
                drug_sensitivity = self.drug_list[i].sensitivity(y, parameters[i])
                r += (dose_combination[i] / s) * drug_sensitivity

            return r

        # initial condition
        if self.drug_list[0].monotone_increasing:
            y0 = self.drug_list[0].control_response + 1e-4
        else:
            y0 = self.drug_list[0].control_response - 1e-4
        # timepoints
        t = np.array([0, s])
        # solve ode
        solution = odeint(f, y0, t, args=(dose_combination,))

        if not gradient:
            return solution[-1][0]  # returns last element in array
        """
        gradient using finite differences
        """

        def fv(effect,
               t,
               dose_combination: np.array,
               v: np.array):

            """
            ith summand of right hand side of hand ODE with changed parameters for finite differences
            """
            r = 0

            for i in range(len(self.drug_list)):
                p = np.array([parameters[i][0] + v[i][0], parameters[i][1] + v[i][1], parameters[i][2] + v[i][2]])
                for j in range(3):
                    if p[j] < 0:
                        p[j] = 1e-4
                drug_sensitivity = self.drug_list[i].sensitivity(effect, p)
                r += (dose_combination[i] / s) * drug_sensitivity

            return r

        # the gradient only works for two drugs

        grad = np.array([])
        for i in range(len(self.drug_list)):
            for j in range(3):
                off = 0.1

                if parameters[i][j] < 2 * off:
                    off = 0

                if i == 0:
                    if j == 0:
                        v = np.array([[off, 0, 0], [0, 0, 0]])
                    if j == 1:
                        v = np.array([[0, off, 0], [0, 0, 0]])
                    if j == 2:
                        v = np.array([[0, 0, off], [0, 0, 0]])

                if i == 1:
                    if j == 0:
                        v = np.array([[0, 0, 0], [off, 0, 0]])
                    if j == 1:
                        v = np.array([[0, 0, 0], [0, off, 0]])
                    if j == 2:
                        v = np.array([[0, 0, 0], [0, 0, off]])

                y1 = odeint(fv, y0, t, args=(dose_combination, v))
                y2 = odeint(fv, y0, t, args=(dose_combination, -v))

                if not off == 0:
                    grad = np.append(grad, (y1[-1][0] - y2[-1][0]) / (2 * off))
                else:
                    grad = np.append(grad, parameters[i][j])

        return solution[-1][-1], grad

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
        number_of_drugs = len(self.drug_list)

        if parameters is None:
            parameters = [self.drug_list[i].parameters for i in range(number_of_drugs)]

        responses = [self.drug_list[i].get_response(dose_combination[i]) for i in range(number_of_drugs)]

        if self.drug_list[0].monotone_increasing:
            hsa = np.argmax(responses)
        else:
            hsa = np.argmin(responses)

        if gradient:
            (response, grad_of_hsa) = self.drug_list[hsa].get_response(dose_combination[hsa], parameters[hsa], True)
            grad = np.zeros((number_of_drugs, 3))
            grad[hsa] = grad_of_hsa
            return response, self._matrix_to_vector(grad)
        else:
            return responses[hsa]

    def evaluate_rss_single_drug_data(self,
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
        log-likelihood: float

        grad: np.array
            gradient of log-likelihood

        """
        if not gradient:
            sum_of_residuals = np.sum([drug.evaluate_lsq_residual(drug.parameters, False)
                                       for drug in self.drug_list])
            return sum_of_residuals / self.sigma2

        else:
            number_of_drugs = len(self.drug_list)
            (sum_of_residuals, grad) = self.drug_list[0].evaluate_lsq_residual(parameters[0], True)

            for i in range(1, number_of_drugs):
                (lsq_new, grad_new) = self.drug_list[i].evaluate_lsq_residual(parameters[i], True)
                sum_of_residuals += lsq_new
                grad = np.append(grad, grad_new)

            return sum_of_residuals / self.sigma2, grad / self.sigma2

    def evaluate_validation_residual(self,
                                     validation_responses_mean: float,
                                     validation_doses: np.array,
                                     gradient: bool,
                                     parameters: np.array = None,  # 2dim array
                                     null_model: str = 'bliss',
                                     number_of_responses: int = None):
        """
        Calculates the squared residual of validation data point. Also returns the gradient if wanted.

        Parameters
        ----------
        validation_responses_mean: float
            measured response of validation experiment

        validation_doses: np.array
            dose of validation experiment

        gradient: bool
            determines whether gradient should be returned as well

        parameters: np.array = None
            parameters for drugs

        null_model: str
            null_model that is used

        number_of_responses: int
            the number of validation responses

        Returns
        -------
        residual: float
            the calculated squared residual of validation experiment

        grad: np.array
            the gradient of sqaured residual of validation experiment containing partial derivatives for parameters

        """
        get_combination_response = self.combination_response(null_model)

        if parameters is None:
            parameters = [drug.parameters for drug in self.drug_list]

        if gradient:
            (response, grad_prep) = get_combination_response(validation_doses, True, parameters)
            residual = (validation_responses_mean - response) ** 2 / (self.sigma2 / number_of_responses)
            grad = 2 / (self.sigma2 / number_of_responses) * (response - validation_responses_mean) * grad_prep
            return residual, grad
        else:
            return (validation_responses_mean - get_combination_response(validation_doses, False, parameters)) ** 2 / \
                   (self.sigma2 / number_of_responses)

    def fit_to_full_data(self,
                         validation_responses_mean: float,
                         validation_doses: np.array,
                         null_model: str = 'bliss',
                         number_of_responses: int = None,
                         minimum_value: float = None):
        """
        Fits drug parameters to data including validation experiment.

        Parameters
        ----------
        validation_responses_mean: float
            response data of validation experiment

        validation_doses: np.array
            dose_combination for validation experiment

        null_model: str
            null model that is used

        number_of_responses: int
            the number of validation responses

        minimum_value: float
            result before optimization

        Returns
        -------
        minimum_value: float
              minimal value of -2LL for given data
        """
        gradient = True

        def min2loglikelihood(parameters: np.array,  # parameters is array of length 3 * len(drug_list)
                              null_model: str):
            parameters = self._vector_to_matrix(parameters)

            if gradient:
                (r1, grad1) = self.evaluate_rss_single_drug_data(parameters, True)
                (r2, grad2) = self.evaluate_validation_residual(validation_responses_mean, validation_doses, True,
                                                                parameters,
                                                                null_model, number_of_responses)
                return r1 + r2, grad1 + grad2
            else:
                r1 = self.evaluate_rss_single_drug_data(parameters, False)
                r2 = self.evaluate_validation_residual(validation_responses_mean, validation_doses, False,
                                                       parameters,
                                                       null_model, number_of_responses)
            return r1 + r2

        bounds = numpy.matlib.repmat(np.array([(1e-4, 10), (1e-4, 20), (1e-4, 0.99)]), len(self.drug_list), 1)

        minimum_parameters = self._drug_list_to_parameters()

        solution = minimize(min2loglikelihood, minimum_parameters, args=null_model, method='TNC', jac=gradient,
                            bounds=bounds)

        if solution.fun < minimum_value:
            minimum_value = solution.fun

        return minimum_value

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

    def combination_response(self,
                             null_model: str = 'bliss'):
        """
        Returns the null model response function belonging to the given null model.

        Parameters
        ----------
        null_model: str
            the chosen null model

        Returns
        -------
        get_combination_response
            the response function for null_model
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

        return get_combination_response

    def old_fit_volume_likelihood(self,
                                  dose_combination,
                                  combination_responses,
                                  parameters: np.array = None,
                                  null_model: str = 'bliss',
                                  gradient_flag: bool = False):
        """
        Calculates the likelihood using the volume statistic regarding the drug parameters fitted from single drug data.

        The sum statistic means that we want to minimize the sum of validation_measurement - prediction. Hence, we
        approximate the volume between the measurement plane and the prediction plane.

        Parameters
        ----------
        dose_combination:
            the dose combination of the drugs

        combination_responses:
            the measured responses of the combination

        parameters:
            the drug parameters of the combination

        null_model:
            the chosen null model

        gradient_flag: bool
            determines whether gradient should be returned as well

        Returns
        -------
        likelihood:
            the calculated likelihood

        gradient:
            the gradient of the likelihood

        """
        if parameters is None:
            parameters = [drug.parameters for drug in self.drug_list]

        parameters_matrix = self._vector_to_matrix(parameters)
        get_combination_response = self.combination_response(null_model)
        number_of_validation_points = len(combination_responses)
        sum_of_comb_predictions = 0
        sum_of_comb_responses = 0

        sum_of_comb_responses += np.sum(combination_responses)

        if not gradient_flag:
            for i in range(number_of_validation_points):
                combination_prediction = get_combination_response(dose_combination[:, i], False, parameters_matrix)
                sum_of_comb_predictions += combination_prediction

            single_drug_residual = self.evaluate_rss_single_drug_data(parameters_matrix, False)
            likelihood = (sum_of_comb_responses - sum_of_comb_predictions) ** 2 / \
                         (number_of_validation_points * self.sigma2) + single_drug_residual

            return likelihood

        else:
            sum_of_gradients = np.zeros(3 * len(self.drug_list))
            single_drug_residual, grad_single_drugs = self.evaluate_rss_single_drug_data(parameters_matrix, True)
            likelihood = (sum_of_comb_responses - sum_of_comb_predictions) ** 2 / \
                         (number_of_validation_points * self.sigma2) + single_drug_residual

            for i in range(number_of_validation_points):
                (combination_prediction, grad_comb_i) = get_combination_response(dose_combination[:, i], True,
                                                                                 parameters_matrix)
                sum_of_comb_predictions += combination_prediction
                sum_of_gradients += grad_comb_i

            gradient = - 2 / (number_of_validation_points * self.sigma2) * (
                    sum_of_comb_responses - sum_of_comb_predictions) * sum_of_gradients + grad_single_drugs

            return likelihood, gradient

    def new_fit_volume_likelihood(self,
                                  dose_combination,
                                  combination_responses,  # only implemented for one response
                                  null_model: str = 'bliss'):
        """
        Optimizes the sum likelihood.

        Parameters
        ----------
        dose_combination:
            the dose combination for the drugs

        combination_responses:
            the measured response of the combination experiment

        null_model:
            the chosen null model

        Returns
        -------
        solution.fun:
            the optimized sum likelihood
        """

        def min2loglikelihood(parameters: np.array,  # parameters is array of length 3 * len(drug_list)
                              null_model: str):
            return self.old_fit_volume_likelihood(dose_combination, combination_responses, parameters, null_model)


        bounds = numpy.matlib.repmat(np.array([(1e-8, 10), (1e-8, 10), (1e-8, 0.99)]), len(self.drug_list), 1)
        initial_parameters = self._drug_list_to_parameters()
        solution = minimize(min2loglikelihood, initial_parameters, args=null_model, method='TNC', jac=False,
                            bounds=bounds)
        return solution.fun

    def volume_significance(self,
                            dose_combinations,
                            responses,
                            null_model: str = 'bliss'):
        """
        Computes the significance level for a given dose and measurement and null model using the sum statistic.

        Parameters
        ----------
        dose_combinations: np.array
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
        old = self.old_fit_volume_likelihood(dose_combinations, responses, None, null_model)
        new = self.new_fit_volume_likelihood(dose_combinations, responses, null_model)
        difference = old - new
        return chi2.sf(difference, 1, loc=0, scale=1)

    def get_significance(self,
                         dose_combination: np.array,
                         responses: np.array,
                         null_model: str = 'bliss'):
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
        if isinstance(responses, float):
            number_of_responses = 1
        else:
            number_of_responses = len(responses)

        responses = np.mean(responses)
        theta_y = [drug.parameters for drug in self.drug_list]
        rss_y_y = self.evaluate_rss_single_drug_data(theta_y, False)
        rss_y_yz = rss_y_y + self.evaluate_validation_residual(responses, dose_combination, False,
                                                               self._vector_to_matrix(theta_y), null_model,
                                                               number_of_responses)
        rss_yz_yz = self.fit_to_full_data(responses, dose_combination, null_model, number_of_responses, rss_y_yz)

        difference = rss_y_yz - rss_yz_yz
        sig = chi2.sf(difference, 1, loc=0, scale=1)
        return sig

    def _check_drug_consistency(self,
                                drug_list: list) -> bool:
        """
        check, if all drugs are either mon incr or decr. ...
        check, if all drugs share the same w_0 ...
        If yes return True, if not return False.
        """
        number_of_drugs = len(drug_list)
        control_response = drug_list[number_of_drugs - 1].control_response
        monotony_flag = drug_list[number_of_drugs - 1].monotone_increasing

        for i in range(number_of_drugs - 1):
            if not (drug_list[i].control_response == control_response):
                return False
            if not (drug_list[i].monotone_increasing == monotony_flag):
                return False
        return True

    def _set_sigma2(self):
        """
        Compute the (optimal) sigma for the given drugs in drug_list using hierarchical optimization.

        """
        number_of_drugs = len(self.drug_list)
        sum_of_variances = 0
        for i in range(number_of_drugs):
            sum_of_variances += self.drug_list[i].get_sigma2()
        self.sigma2 = sum_of_variances / number_of_drugs
        return
