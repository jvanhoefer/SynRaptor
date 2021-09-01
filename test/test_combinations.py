"""
Tests for the combination models.

When implementing a new CombinationModel, one needs to implement a
`test_<new_model>` function, that checks `get_combined_effect` function of
the new model and add the new model to COMBINATION_MODELS. All other tests
(gradients, etc.) then run automatically.
"""

import unittest
from itertools import product, combinations

import numpy as np
import scipy.optimize as opt

from synraptor import HillCurveModel
from synraptor import (CombinationModelBase,
                       HSACombinationModel,
                       BlissCombinationModel)

COMBINATION_MODELS = [HSACombinationModel,
                      BlissCombinationModel]


class TestCombinations(unittest.TestCase):
    """
    TestCase class for testing all functionality from the Combinations class.
    """

    def setUp(self):
        # test parameters
        self.test_parameters = [np.array([3, 2, 1]),
                                np.array([0.5, 0.5, 1])]

    def test_get_index_range(self):
        """
        Tests the get_index_range function.
        """
        drug_list = [HillCurveModel(), HillCurveModel()]
        test_combination = HSACombinationModel(drug_list)

        self.assertTupleEqual(test_combination.get_index_range(0), (0, 3))
        self.assertTupleEqual(test_combination.get_index_range(1), (3, 6))

    def test_sanity_checks(self):
        """
        Tests the sanity checks for compatibility of different drugs.
        """
        parameters = np.array([1, 1, 0.5])

        # Test sanity checks for the increasing/decreasing flag
        drug_list = [HillCurveModel(monotone_increasing=True),
                     HillCurveModel(monotone_increasing=False)]

        for i in range(2):
            drug_list[i].parameters = parameters

        for combination_model in COMBINATION_MODELS:
            self.assertRaises(ValueError, combination_model, drug_list)

        # Test sanity checks for control response
        drug_list = [HillCurveModel(monotone_increasing=True,
                                    control_response=0),
                     HillCurveModel(monotone_increasing=False,
                                    control_response=0.1)]
        for i in range(2):
            drug_list[i].parameters = parameters

        for combination_model in COMBINATION_MODELS:
            self.assertRaises(ValueError, combination_model, drug_list)

    def test_bliss_combined_effect(self):
        """
        Tests the combined effects of the Bliss combination
        model.

        Gradient checks are performed in another test.
        """
        for inc, p0, p1 in product([True, False],
                                   self.test_parameters,
                                   self.test_parameters):

            drug_list = _get_drug_list(inc, p0, p1)

            combination = BlissCombinationModel(drug_list)

            # test for zero dose
            control_response = int(not inc)
            self.assertAlmostEqual(
                combination.get_combined_effect([0, 0]),
                control_response,
                places=6)

            # test for half max
            if inc:
                expected_result = 0.5 * (p0[2] + p1[2]) - 0.25 * p0[2] * p1[2]
            else:
                expected_result = (1 - 0.5 * p0[2])*(1 - 0.5 * p1[2])

            self.assertAlmostEqual(
                combination.get_combined_effect([p0[0], p1[0]]),
                expected_result,
                places=6)

            # test comparison to f_A + f_B - f_A*f_B formula,
            # (which is not used in the implementation)
            if inc:
                for d0, d1 in combinations([0, 10, 20, 50, 100], 2):
                    f_A = drug_list[0].get_response(d0)
                    f_B = drug_list[1].get_response(d1)
                    expected_result = f_A + f_B - f_A*f_B

                    self.assertAlmostEqual(
                        combination.get_combined_effect([d0, d1]),
                        expected_result,
                        places=6)

    def test_hsa_combined_effect(self):
        """
        Tests the combined effects (and gradients )of the HSA combination
        model.

        Gradient checks are performed in another test.
        """
        p0 = np.array([1, 2, 1])
        p1 = np.array([1, 2, 1])

        for inc in [True, False]:

            drug_list = _get_drug_list(inc, p0, p1)

            combination = HSACombinationModel(drug_list)

            # both parameters the same...
            for d in [0, 10, 100]:
                self.assertAlmostEqual(
                    combination.get_combined_effect([d, d]),
                    combination.drug_list[0].get_response(d))

            # test if everything works if they are not...
            combination.drug_list[0].parameters = np.array([1, 2, 6])

            for d in [0, 10, 100]:

                self.assertAlmostEqual(
                    combination.get_combined_effect([d, d]),
                    combination.drug_list[0].get_response(d))

    def perform_test_get_combined_effect(
        self,
        combination: CombinationModelBase,
        p_eval: np.array
    ):
        """
        Helper function, that actually performs the tests of the
        get_combined_effect function (+ gradients).
        """
        for doses in [[0, 5], [5, 5]]:

            # test if function value with and without gradient is the same

            f_val_no_grad = \
                combination.get_combined_effect(dose_combination=doses,
                                                parameters=p_eval)
            f_val_w_grad = \
                combination.get_combined_effect(dose_combination=doses,
                                                parameters=p_eval,
                                                gradient=True)[0]
            self.assertAlmostEqual(f_val_no_grad, f_val_w_grad, places=6)

            # perform gradient check
            def f(p):
                return combination.get_combined_effect(dose_combination=doses,
                                                       parameters=p)

            def grad(p):
                return combination.get_combined_effect(dose_combination=doses,
                                                       parameters=p,
                                                       gradient=True)[1]

            difference = opt.check_grad(f, grad, p_eval)
            self.assertAlmostEqual(difference, 0, 3)

    def perform_test_get_single_drug_data_nllh(
        self,
        combination: CombinationModelBase,
        p_eval: np.array
    ):
        """
        Helper function, that actually performs the tests of the
        get_combined_effect function (+ gradients).
        """

        # test if function value is the same with and without gradient
        # computation.

        f_val_no_grad = \
            combination._get_single_drug_data_nllh(parameters=p_eval)

        f_val_w_grad = \
            combination._get_single_drug_data_nllh(parameters=p_eval,
                                                   gradient=True)[0]

        self.assertAlmostEqual(f_val_no_grad, f_val_w_grad, places=6)

        # Check gradients.
        def f(p):
            return combination._get_single_drug_data_nllh(parameters=p)

        def grad(p):
            return combination._get_single_drug_data_nllh(parameters=p,
                                                          gradient=True)[1]

        self.assertAlmostEqual(opt.check_grad(f, grad, p_eval), 0, 3)

    def perform_test_get_full_data_nllh(
            self,
            combination: CombinationModelBase,
            p_eval: np.array
    ):
        """
        Helper function, that actually performs the tests of the
        get_combined_effect function (+ gradients).
        """
        # test for single and multiple responses.
        for doses, response in product([[0, 5], [5, 5]],
                                       [4.0, [4.0, 4.1]]):

            # test function value is the same with and without gradient.
            f_val_no_grad = \
                combination._get_full_data_nllh(dose_combination=doses,
                                                response_data=response,
                                                parameters=p_eval)
            f_val_w_grad = \
                combination._get_full_data_nllh(dose_combination=doses,
                                                response_data=response,
                                                parameters=p_eval,
                                                gradient=True)[0]
            self.assertAlmostEqual(f_val_no_grad, f_val_w_grad, places=6)

            # perform gradient check
            def f(p):
                return combination._get_full_data_nllh(dose_combination=doses,
                                                       response_data=response,
                                                       parameters=p)

            def grad(p):
                return combination._get_full_data_nllh(dose_combination=doses,
                                                       response_data=response,
                                                       parameters=p,
                                                       gradient=True)[1]

            self.assertAlmostEqual(opt.check_grad(f, grad, p_eval), 0, 3)

            # test, if the full data nllh is actually larger then single-drug data nllh

            nllh_full_data = f(p_eval)
            nllh_single_drug_data = \
                combination._get_single_drug_data_nllh(p_eval,
                                                       gradient=False)

            self.assertLess(nllh_single_drug_data, nllh_full_data)

    def test_gradients(self):
        """
        Test _get_single_drug_data_nllh and _get_full_data_nllh for all models.
        Additionally tests the gradients of these methods...
        :return:
        """

        for inc, p0, p1, combination_model \
                in product([True, False],
                           self.test_parameters,
                           self.test_parameters,
                           COMBINATION_MODELS):

            drug_list = _get_drug_list(inc, p0, p1)

            # add drug measurement data (+ 0.1 = "measurement noise")
            for i, drug in enumerate(drug_list):
                drug_list[i].dose_data = np.arange(10)
                # add measurement error
                drug_list[i].response_data = \
                    drug.get_multiple_responses(np.arange(10)) + 0.1

            combination = combination_model(drug_list)

            # test if the function values with an without gradients
            # are the same...
            for p_eval_0, p_eval_1 in combinations(self.test_parameters, 2):

                p_eval = np.concatenate((p_eval_0, p_eval_1))

                self.perform_test_get_single_drug_data_nllh(combination, p_eval)
                self.perform_test_get_combined_effect(combination, p_eval)
                self.perform_test_get_full_data_nllh(combination, p_eval)

    def test_get_likelihood_ratio_significance(self):
        """
        Tests the get_likelihood_significance function.
        """

        # set up test combinations
        for inc, p0, p1, combination_model \
                in product([True, False],
                           self.test_parameters,
                           self.test_parameters,
                           COMBINATION_MODELS):

            drug_list = _get_drug_list(inc, p0, p1)

            # add drug measurement data (+ 0.1 = "measurement noise")
            for i, drug in enumerate(drug_list):
                drug_list[i].dose_data = np.arange(10)
                # add measurement error
                drug_list[i].response_data = \
                    drug.get_multiple_responses(np.arange(10)) + 0.1

                # fit parameters...
                drug_list[i]._set_fitting_problem()
                drug_list[i].fit_parameters()

            combination = combination_model(drug_list)

            for d0, d1 in combinations([0, 3, 10], 2):

                # test if the significance of the predicted response is 1
                # (for single and multiple combination experiments)

                doses = [d0, d1]
                expected_response = \
                    combination.get_combined_effect(dose_combination=doses)

                # test for single response
                self.assertAlmostEqual(
                    combination.get_likelihood_ratio_significance(
                        doses, expected_response),
                    1,
                    places=3)

                # test for multiple responses
                self.assertAlmostEqual(
                    combination.get_likelihood_ratio_significance(
                        doses, [expected_response, expected_response]),
                    1,
                    places=3)

                # test if the significances drop, when the deviation from the
                # predicted response increases

                # significance from response + delta from previous iteration...
                significance_plus = 1
                significance_minus = 1

                for delta_response in [0.1, 1, 5, 10]:

                    sig_test_plus = \
                        combination.get_likelihood_ratio_significance(
                            doses, expected_response + delta_response)

                    sig_test_minus = \
                        combination.get_likelihood_ratio_significance(
                            doses, expected_response - delta_response)

                    # self.assertLessEqual(sig_test_plus, significance_plus)
                    self.assertLessEqual(sig_test_minus, significance_minus)

                    significance_plus = sig_test_plus
                    significance_minus = sig_test_minus

                # test if the last significances are close to zero
                # self.assertAlmostEqual(sig_test_plus, 0, places=5)
                self.assertAlmostEqual(sig_test_minus, 0, places=5)

    def test_get_likelihood_ratio_cis(self):
        """
        Tests the get_likelihood_ratio_ci function.
        """

        for inc, p0, p1, combination_model \
                in product([True, False],
                           self.test_parameters,
                           self.test_parameters,
                           COMBINATION_MODELS):

            drug_list = _get_drug_list(inc, p0, p1)

            # add drug measurement data (+ 0.1 = "measurement noise")
            for i, drug in enumerate(drug_list):
                drug_list[i].dose_data = np.arange(10)
                # add measurement error
                drug_list[i].response_data = \
                    drug.get_multiple_responses(np.arange(10)) + 0.1

                # fit parameters...
                drug_list[i]._set_fitting_problem()
                drug_list[i].fit_parameters()

            combination = combination_model(drug_list)

            for d0, d1 in combinations([0, 3, 10], 2):

                # test if the significance of the lower/upper bounds
                # are equal to 1-alpha.

                doses = [d0, d1]
                for alpha in [0.5, 0.8, 0.95]:
                    ci = combination.get_likelihood_ratio_ci(doses,
                                                             alpha,
                                                             a_tol=1e-4)
                    alpha_emp = [1 - combination.get_likelihood_ratio_significance(doses, ci[0]),
                                 1 - combination.get_likelihood_ratio_significance(doses, ci[1])]

                    if np.isfinite(ci[0]):
                        self.assertAlmostEqual(
                            alpha_emp[0],
                            alpha,
                            places=2)
                    if np.isfinite(ci[1]):
                        self.assertAlmostEqual(
                            alpha_emp[1],
                            alpha,
                            places=2)

    def test_get_sampling_predictions(self):
        raise NotImplementedError("Not implemented yet.")

    def test_get_sampling_significance(self):
        raise NotImplementedError("Not implemented yet.")

    def get_sampling_ci(self):
        raise NotImplementedError("Not implemented yet.")


def _get_drug_list(inc: bool,
                   p0: np.ndarray,
                   p1: np.ndarray):
    """
    Function returns a drug list with two entries.
    """

    # control_response = int(not inc) => 1 if decreasing and 0 else
    control_response = int(not inc)

    drug_list = [HillCurveModel(monotone_increasing=inc,
                                control_response=control_response),
                 HillCurveModel(monotone_increasing=inc,
                                control_response=control_response)]

    drug_list[0].parameters = p0
    drug_list[1].parameters = p1

    return drug_list

