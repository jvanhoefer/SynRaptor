import unittest

import numpy as np
import scipy.optimize as opt

from synraptor import HillCurveModel


class TestDrug(unittest.TestCase):
    """
    TestCase class for testing all functionality from the DoseResponseModel class
    """

    def test_increasing(self):
        """
        Test, if the monotone increasing flag is used correctly,

        i.e. that drugs with monotone_increasing True/False are monotone increasing/
        decreasing correspondingly.
        """

        for increasing in [True, False]:
            test_drug = HillCurveModel(monotone_increasing=increasing)
            test_drug.parameters = np.array([2, 5, 1])

            if increasing:
                self.assertGreater(test_drug.get_response(5), test_drug.get_response(0))

            else:
                self.assertLess(test_drug.get_response(5), test_drug.get_response(0))

    def test_evaluate_dose_response(self):
        """
        Test special values for dose response values (half max and no effect...)
        """
        # Zero Dose:
        # ----------
        w_0 = 0.5
        test_drug = HillCurveModel(monotone_increasing=True,
                                   control_response=w_0)
        test_drug.parameters = np.array([3, 2, 1])

        self.assertAlmostEqual(test_drug.get_response(0), w_0)

        # Half max:
        # ---------
        # Due to the parameter choice we expect (a=2, n=2)
        # to be the halve max. This is tested.

        test_drug = HillCurveModel(monotone_increasing=True)
        test_drug.parameters = np.array([2, 2, 1])

        self.assertAlmostEqual(test_drug.get_response(2), 0.5)

    def test_evaluate_multiple_responses(self):
        """
        Test if multiple_responses give the same result as single ones.
        """

        test_drug = HillCurveModel(monotone_increasing=True,
                                   control_response=0)

        parameters = np.array([3, 2, 1])

        expected_results = np.nan * np.ones(100)

        for i in range(100):
            expected_results[i] = test_drug.get_response(i, parameters=parameters)

        results = test_drug.get_multiple_responses(np.arange(100), parameters)

        np.testing.assert_almost_equal(expected_results, results)

    def test_gradient_response(self):
        """
        Gradient checks for single response function.

        checks for different doses AND parameters AND increasing True/False
        """
        increasing = [True, False]
        doses = [0, 5, 10, 20, 50, 100]
        parameter_sets = [np.array([3, 2, 1]),
                          np.array([1, 1, 1]),
                          np.array([3, 2, 5])]

        for (inc, dose, parameters) in zip(increasing, doses, parameter_sets):

            test_drug = HillCurveModel(monotone_increasing=inc,
                                       control_response=0)

            def f(p):
                r = test_drug.get_response(dose, p, False)
                return test_drug.get_response(dose, p, False)

            def grad(p):
                r = test_drug.get_response(dose, p, True)[1]
                return test_drug.get_response(dose, p, True)[1]

            difference = opt.check_grad(f, grad, parameters)

            # CAUTION: If this test fails, maybe the "3" below is a bit to strict,
            # so maybe relax this to 2, or even 1
            self.assertAlmostEqual(difference, 0, 3)

    def test_evaluate_lsq_residuals(self):
        """
        Tests, if the LSQ residuals are reasonable.
        """
        parameters = np.array([3, 2, 1])

        test_drug = HillCurveModel(monotone_increasing=True,
                                   control_response=0)
        test_drug.parameters = parameters

        n_doses = 100
        doses = np.arange(n_doses)
        response = test_drug.get_multiple_responses(doses)

        for residual in [0, 1, 2]:
            # via response+residual we now the (pointwise) residual
            test_drug._set_dose_and_response(doses, response+residual)
            expected_lsq_residual = residual**2 * n_doses
            obtained_lsq_residual = test_drug.evaluate_lsq_residual(parameters)

            self.assertAlmostEqual(expected_lsq_residual, obtained_lsq_residual)

    def test_gradient_lsq_residuals(self):
        """
        Gradient checks for LSQ residuals function.

        checks for different doses AND parameters AND increasing True/False
        """
        increasing = [True, False]
        doses = np.array([0, 5, 10, 20, 50, 100])
        parameter_sets = [np.array([3, 2, 1]),
                          np.array([1, 1, 1]),
                          np.array([3, 2, 5])]

        for (inc, parameters) in zip(increasing, parameter_sets):

            test_drug = HillCurveModel(monotone_increasing=inc,
                                       control_response=0)
            responses = test_drug.get_multiple_responses(doses=doses,
                                                         parameters=parameters,
                                                         gradient=False)

            # set doses and responses and define function+gradient
            test_drug._set_dose_and_response(doses, responses)

            def f(p):
                return test_drug.evaluate_lsq_residual(p, False)

            def grad(p):
                return test_drug.evaluate_lsq_residual(p, True)[1]

            difference = opt.check_grad(f, grad, parameters)

            self.assertAlmostEqual(difference, 0, 3)

    def test_fitting(self):
        """
        Tests the fitting procedure.

        We test, if for noise free data the (pointwise) residuals are close to zero...
        """

        test_parameters = np.array([5, 2, 2])

        test_drug = HillCurveModel(monotone_increasing=True,
                                   control_response=0)
        test_drug.parameters = test_parameters

        n_doses = 100
        doses = np.arange(n_doses)
        synthetic_response = test_drug.get_multiple_responses(doses)
        test_drug = HillCurveModel(dose_data=doses,
                                   response_data=synthetic_response,
                                   monotone_increasing=True,
                                   control_response=0,
                                   lb=np.array([0, 0, 0]),
                                   ub=np.array([10, 10, 10]))

        # Perform fitting
        test_drug.fit_parameters()
        fitted_response = test_drug.get_multiple_responses(doses)

        self.assertTrue(np.max(np.abs(synthetic_response - fitted_response))
                        < 0.1)
