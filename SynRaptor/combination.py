"""Combinations of multiple drugs are defined here.
"""
import numpy as np
import scipy as sp
from SynRaptor import drug
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import pandas as pd





class Combination:
    """
    Combination stores a list of drugs and gives all functionality
    to evaluate synergy null models and derive synergy.

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
        Compute the Loewe response.

        Uses implicit functions theorem to calculate the gradient if required.

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
        effect: float
            the response of the combined drug

        grad: np.array
            the gradient
        """


        l = len(dose_combination)
        if parameters is None:
            parameters = [self.drug_list[i].parameters for i in range(l)]

        def F(effect):
            s = sum(dose_combination[i] / self.drug_list[i].inverse_evaluate(effect, parameters[i]) for i in range(l))
            return s - 1

        effect = sp.optimize.bisect(F, 0, 9)

        if not gradient:
            return effect

        def get_parameters(i: int):
            return self.drug_list[i].control_response, parameters[i][0], parameters[i][1], parameters[i][2]


        def div1y(y,
                 i: int): #index of drug in drug_list
            """
            Calculates the derivative of 1/f_i^{-1}(y)
            """
            w,a,n,s = get_parameters(i)
            return - s * a **n*(a**n*(y-w)/(s+w-y))**(-1/(n-1))/(n*(s+w-y)**2)


        divFy = sum(dose_combination[i] * div1y(effect,i) for i in range(l)) #calculates dF/dy

        def divYa(y,
                  i: int):
            """
            Calculates dy/da_i
            """
            w, a, n, s = get_parameters(i)
            return dose_combination[i]*(a**n*(y-w)/(s+w-y))**(-1/n)/a / divFy


        def divYn(y,
                  i: int):
            """
            Calculates dy/dn_i
            """
            w,a,n,s = get_parameters(i)
            return dose_combination[i] * ((y-w)/(s+w-y))**(-1/n)*math.log((y-w)/(s+w-y))/(a*n**2) / divFy #a,n,w,s,y are positive


        def divYs(y,
                  i: int):
            """
            Calculates dy/ds_i
            """
            w,a,n,s = get_parameters(i)
            return dose_combination[i] *(a**n*(y-w)/(s+w-y))**(-1/n)/(n*(s+w-y)) / divFy


        grad = [np.array([divYa(effect,i),divYn(effect,i),divYs(effect,i)]) for i in range(l)]

        return effect, self.matrix_to_vector(grad)


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

        l = len(dose_combination)
        if parameters is None:
            parameters = [self.drug_list[i].parameters for i in range(l)]
        self._check_bliss_requirements(parameters)

        # For monotone increasing drugs the Bliss response is 1-prod(1-y_i)
        if self.drug_list[0].monotone_increasing:
            if not gradient:
                return 1 - np.prod([1 - self.drug_list[i].get_response(dose_combination[i], parameters[i]) for i in range(l)])

            matrix = [self.drug_list[i].get_response(dose_combination[i], parameters[i], True) for i in range(l)]
            oneminusresponses = [1 - matrix[i][0] for i in range(l)]
            prod = np.prod(oneminusresponses)
            grad = [prod / (oneminusresponses[i]) * matrix[i][1] for i in range(l)]
            grad = self.matrix_to_vector(grad)  # now gradient looks like [a0 n0 s0 a1 n1 s1 ...]
            return 1 - prod, grad
        # For monotone decreasing drugs the Bliss response is prod(y_i)
        else:
            if not gradient:
                return np.prod([self.drug_list[i].get_response(dose_combination[i], parameters[i]) for i in range(l)])

            matrix = [self.drug_list[i].get_response(dose_combination[i], parameters[i], True) for i in range(l)]
            responses = [matrix[i][0] for i in range(l)]
            prod = np.prod(responses)
            grad = [prod / responses[i] * matrix[i][1] for i in range(l)]
            grad = self.matrix_to_vector(grad)  # now gradient looks like [a0 n0 s0 a1 n1 s1 ...]
            return prod, grad


    def _check_bliss_requirements(self,
                                  parameters: np.array):
        """
        Checks if requirements for get_bliss_response are fulfilled.

        Checks if for monotone increasing drugs the control_response is 0, and the maximal effect is <= 1. As the
        control response is the same for all drugs it it sufficient to check for drug_list[0].
        For decreasing drugs the control_response has to be 1 and the parameter s of Hill curve <=1. If one requirement
        is not met, False is returned. Otherwise True is returned.
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


    def get_multiple_bliss_responses(self,
                                     dose_combinations: np.array,#2dimensional
                                     gradient: bool):
        """
        Computes multiple Bliss responses using get_bliss_response. Also returns the gradient if wanted.

        Parameters
        ----------
        dose_combinations: np.array
            two dimensional array with dose_combinations[i] containing a dose or each drug in self.drug_list

        gradient: bool
            determines wether gradients should be returned as well

        Returns
        -------
        responses: np.array
            responses according to Bliss

        grad: np.array
            two dimensional array so that the ith coloumn corresponds to gradient of bliss at dose_combinations[i]
        """
        l = len(dose_combinations)
        responses = np.nan * np.ones(l)
        if not gradient:
            for i in range(l):
                responses[i] = self.get_bliss_response(dose_combinations[i], False)
            return responses
        l2 = len(dose_combinations[0])
        grad = np.nan * np.ones((l, 3 * l2))
        for i in range(l):
            (responses[i], grad[i]) = self.get_bliss_response(dose_combinations[i], True)#row of grad looks like [dda0 ddn0 dds0 dda1 ddn1 dds1 ...].]
        grad = np.transpose(grad)#coloumn looks like [dda0 ddn0 dds0 dda1 ddn1 dds1 ...]
        return responses, grad


    def evaluate_log_likelihood(self,
                                parameters: np.array,#2dim matrix
                                gradient: bool):
        """
        Evaluates -2 * log likelihood of data.

        Parameters
        ----------
        parameters: np.array
            parameters of drugs

        gradient: bool
            determines wether gradient should be returned as well

        Returns
        -------
        loglikelihood: float

        grad: np.array
            gradient of loglikelihood

        """
        if not gradient:
            return np.sum([self.drug_list[i].evaluate_lsq_residual(self.drug_list[i].parameters, False) \
                           for i in range(len(self.drug_list))])

        else:
            l = len(self.drug_list)
            (sum, grad) = self.drug_list[0].evaluate_lsq_residual(parameters[0], True)

            for i in range(1,l):
                (lsq, gradi) = self.drug_list[i].evaluate_lsq_residual(parameters[i], True)
                sum += lsq
                grad = np.append(grad, gradi)

            return sum / self.sigma2, grad


    def evaluate_validation_residual(self,
                                     validation_response,
                                     validation_dose: np.array,
                                     gradient: bool,
                                     parameters: np.array = None,#2dim array
                                     null_model: str='bliss'):
        """
        Calculates the squared residual of validation data point. Also returns the gradient if wanted.

        Parameters
        ----------
        validation_response: float
            measured response of validation experiment

        validation_dose: np.array
            dose of validation experiment

        gradient: bool
            determines wether gradient should be returned as well

        parameters: np.array = None
            parameters for drugs

        null_model: str
            null_model that is used

        Returns
        -------
        residual: float
            the calculated squared residual of validation experiment

        grad: np.array
            the gradient of sqaured residual of validation experiment containing partial derivatives for parameters

        """
        if null_model == 'bliss':
            get_combination_response = self.get_bliss_response
        else:
            if null_model == 'hsa':
                get_combination_response = self.get_hsa_response
            else:
                if null_model == 'loewe':
                    get_combination_response = self.get_loewe_response
                else:
                    ValueError()


        if parameters is None:
            parameters = [self.drug_list[i].parameters for i in range(len(drug_list))]


        if not gradient:
            return ((validation_response - get_combination_response(validation_dose, False, parameters)) / self.sigma2) ** 2
        (response, grad) = get_combination_response(validation_dose, True, parameters)
        residual = ((validation_response - response) / self.sigma2) ** 2
        grad = 2 * grad * (response - validation_response)
        return residual, grad



    def fit_to_full_data(self,
                         validation_response: float,
                         validation_dose: np.array,
                         null_model: str = 'bliss'):
        """
        Fits drug parameters to data including validation experiment.

        Parameters
        ----------
        validation_dose: float
            response data of validation experiment

        validation_dose: np.array
            dose_combination for validation experiment

        null_model: str
            null model that is used

        Returns
        -------
        solution.fun: float
              minimal value of -2LL for given data

        solution.x: np.array
             optimized parameters that minimize -2LL
        """

        def min2loglikelihood(parameters: np.array,#parameters is array of length 3 * len(drug_list)
                              null_model: str):
            l = len(drug_list)
            parameters = self.vector_to_matrix(parameters)
            (r1, grad1) = self.evaluate_log_likelihood(parameters, True)
            (r2, grad2) = self.evaluate_validation_residual(validation_response, validation_dose, True, parameters, null_model)
            return r1 + r2, grad1 + grad2


        l = len(self.drug_list)
        bounds = np.ones((3*l), dtype=(float,2))
        for i in range(l):
            bounds[3*i] = 1e-8, 10
            bounds[3*i+1] = 1e-8,10
            bounds[3*i+2] = 1e-8, 0.99 # for Bliss s has to be between 0 and 1

        initialParameters = self.drug_list_to_parameters()
        solution = minimize(min2loglikelihood, initialParameters, args=(null_model), method='TNC', jac=True, bounds=bounds)
        return solution.fun, solution.x


    def _get_optimization_starts(self,
                                 bounds: np.array,
                                 n_starts: int):
        """
        Computes initial_values for optimization using Latin Hypercube sampling.

        Parameters
        ----------
        bounds: np.array
            parameter bounds for optimization

        n_starts: int
            number of optimization starts

        Returns
        -------
        np.transpose(initial_values): np.array
            initial_values for optimization
        """
        l = len(bounds)
        initial_values = [[0] * n_starts for i in range(l)]
        for i in range(l):
            perm = np.random.permutation(n_starts)
            initial_values[i] = np.array([bounds[i][0] + (bounds[i][1] - bounds[i][0]) /\
                                          n_starts * (perm[j] + 0.5) for j in range(n_starts)])
        return np.transpose(initial_values)

    def matrix_to_vector(self,
                         matrix: np.array):
        """
        Reshapes two dim array into one dim array.
        """
        return np.reshape(matrix, -1 , order='C')

    def vector_to_matrix(self,
                 vector: np.array):
        """
        Reshapes vector to two dim array with len(matrix[i])=3.
        """
        return np.reshape(vector, (-1,3), order='C')



    def parameters_to_drug_list(self,
                                parameters: np.array):
        """
        Saves array of parameters in self.drug_list.
        """
        for i in range(len(self.drug_list)):
            self.drug_list[i].parameters[0] = parameters[3 * i]
            self.drug_list[i].parameters[1] = parameters[3 * i + 1]
            self.drug_list[i].parameters[2] = parameters[3 * i + 2]
        return


    def drug_list_to_parameters(self):
        """
        Creates array containing the parameters of self.drug_list.
        """
        l = len(self.drug_list)
        parameters = np.nan * np.ones(3*l)
        for i in range(l):
            parameters[3 * i] = self.drug_list[i].parameters[0]
            parameters[3 * i + 1] = self.drug_list[i].parameters[1]
            parameters[3 * i + 2] = self.drug_list[i].parameters[2]
        return parameters


    def get_hand_response(self,
                          dose_combination: np.array,
                          gradient: bool):
        """

        Compute the hand response
        """

        s = sum(dose_combination)

        def f(y, t,
                dose_combination: np.array,
              ):
            l = len(self.drug_list)

            r = 0
            for i in range(l):
                dev = self.drug_list[i].get_derivative(self.drug_list[i].inverse_evaluate(y))
                r += (dose_combination[i] / s) * dev
            return r
        # initial condition
        y0 = 1e-7
        # timepoints
        t = np.array([0,s])
        # solve ode
        y = odeint(f,y0,t, args = (dose_combination,))
        if not gradient:
            # wir wollen y[-1] also das letzte element im array
            return y[-1]
        """
        gradient using finite differences
        """
        def fv(y, t,
                dose_combination: np.array,
               i :int,
              v: np.array,
              ):

            p = np.array([self.drug_list[i].parameters[0]+v[0],self.drug_list[i].parameters[1]+v[1],\
                          self.drug_list[i].parameters[2]+v[2]])
            dev = self.drug_list[i].get_derivative(self.drug_list[i].inverse_evaluate(y,p),p)

            return (dose_combination[i] / s) * dev

        grad = np.array([])
        for i in range(len(self.drug_list)):
            for j in range(3):
                v = np.array([0,0,0])
                v[j] = v[j] + 1

                y1 = odeint(fv, y0, t, args=(dose_combination,i,v))
                y2 = odeint(fv,y0,t, args = (dose_combination,i,-v))
                grad = np.append(grad, (y1[-1]-y2[-1])/2)
                print(y1[-1],y2[-1])
        return y[-1], grad




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
            l = len(self.drug_list)
            if parameters is None:
                parameters = [self.drug_list[i].parameters for i in range(l)]

            responses = [drug_list[i].get_response(dose_combination[i]) for i in range(l)]
            if gradient:
                if drug_list[0].monotone_increasing:
                    # monotone increasing and gradient

                    max = np.argmax(responses)
                    (response, gradm) = self.drug_list[max].get_response(dose_combination[max], parameters[max], True)
                    grad = np.zeros((l, 3))
                    grad[max] = gradm
                    return response, self.matrix_to_vector(grad)
                else:
                    # monotone decreasing and gradient

                    min = np.argmin(responses)
                    (response, gradm) = self.drug_list[min].get_response(dose_combination[min], parameters[min], True)
                    grad = np.zeros((l, 3))
                    grad[min] = gradm
                    return response, self.matrix_to_vector(grad)
            else:
                if drug_list[0].monotone_increasing:
                    # monotone increasing without gradient
                    return np.max(responses)
                else:
                    # monotone decreasing without gradient
                    return np.min(responses)


    def get_loewe_significance(self,
                               dose_combination: np.array,
                               responses: np.array):
        """
        Compute the Loewe significance level for a given dose and measurements.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        """
        raise NotImplementedError

    def get_bliss_significance(self,
                               dose_combination: np.array,
                               responses: np.array):
        """
        Compute the Bliss significance level for a given dose and measurements.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        """
        raise NotImplementedError

    def get_hand_significance(self,
                              dose_combination: np.array,
                              responses: np.array):
        """
        Compute the Hand significance level for a given dose and measurements.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        """
        raise NotImplementedError

    def get_hsa_significance(self,
                             dose_combination: np.array,
                             responses: np.array):
        """
        Compute the HSA significance level for a given dose and measurements.

        Here dose has size number_of_drugs and measurements are arbitrarily many msmts
        for the given dose_combination
        """
        raise NotImplementedError

    def _check_drug_consistency(self,
                                drug_list: list)->bool:
        """
        check, if all drugs are either mon incr or decr. ...
        check, if all drugs share the same w_0 ...
        If yes return True, if not return False.
        """
        l = len(drug_list)
        control_response = drug_list[l-1].control_response
        mon = drug_list[l-1].monotone_increasing
        for i in range(l-1):
            if not (drug_list[i].control_response == control_response):
                return False
            if not (drug_list[i].monotone_increasing == mon):
                return False
        return True


    def _set_sigma2(self):#TODO numerisch super instabil
        """
        Compute the (optimal) sigma for the given drugs in drug_list
        #formel von hiererchischer optimierung für alle drugs zusammen
        """
        l = len(self.drug_list)
        sum = 0
        for i in range(l):
            sum += self.drug_list[i].get_sigma2()
        self.sigma2 = sum / l
        return




#This code may be used for testing.
#print('hallo')
x = np.array([1,2,3,4,5])
y = np.array([2,4,6,6,7])
z = np.array([1,2,4,5,7])


A = drug.Drug(x, 0.0001 * y, True, 0)
A.fit_parameters(10)

#print(A.get_derivative(7))
#print(A.inverse_evaluate(7))

B = drug.Drug(y, 0.0001 * z, True, 0)
B.fit_parameters(10)


C = drug.Drug(x, 0.0001 * z, True, 0)
C.fit_parameters(10)

D = drug.Drug(y, 0.0001 * y, True, 0)
D.fit_parameters(10)


A.parameters = np.array([3,2,0.8])
B.parameters= np.array([7,1,0.8])
C.parameters= np.array([3,1.5,0.4])
D.parameters= np.array([4,2,1])

drug_list = [A,B]
Comb = Combination(drug_list)
dose1 = np.array([3,7])
dose2 = np.array([0.6,0.5,0.5,0.2])
dose3 = np.array([0.5,0.3,0.5,0.1])
dosez = np.array([0.5,0.5,0.4,0.6])
doses = np.array([dose1,dose2, dose3])


responses = np.array([0.6,0.8, 0.7])
#print('inverse', A.inverse_evaluate(0.6))

#print('Comb.newbliss:',Comb.newbliss(dosez,True))
#res = Comb.get_hsa_response(dose, True)
#print('Comb.get_bliss_response(dosez, True, None):',Comb.get_bliss_response(dosez, True, None))
#print('Comb.get_hsa_response(dosez, True, None):',Comb.get_hsa_response(dosez, True, None))
#res2 = Comb.get_multiple_bliss_responses(doses, True)
#res4 = Comb.evaluate_log_likelihood(responses,doses, True)
#res3 = Comb.get_hand_response(dose1,True)
#print(res2)
#print(res4)
#print(res3)


#c = Comb.fit_to_full_data(0.7, dosez, 'loewe')

#print(c)

#p = Comb.drug_list_to_parameters()
#print(p)
#Comb.parameters_to_drug_list(p)
#a = Comb.evaluate_validation_residual(0.7,dosez,True,None,'hsa')
#b = Comb.evaluate_log_likelihood(responses,doses)
#print(a)
#print(b)

#print(Comb._set_sigma2())

#print('sigma2:', Comb.sigma2)
print('Loewe:',Comb.get_loewe_response(dose1, True))
print('Hand:', Comb.get_hand_response(dose1, True))


