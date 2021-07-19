"""
Implements the Bliss Combination model.
"""
import numpy as np
from typing import List, Union

from .CombinationModelBase import CombinationModelBase
from ..DoseResponseModels import DoseResponseModelBase


class BlissCombinationModel(CombinationModelBase):
    """
    Implements the Bliss Combination model, defined by


    """
    def __init__(self,
                 drug_list: List[DoseResponseModelBase]):
        """
        Constructor.
        """

        for drug in drug_list:
            # dose response curve not bound to [0, 1]
            if not drug.check_bliss_consistency():
                raise ValueError("Dose response curves need to be bound to "
                                 "[0, 1] in order for the Bliss Combination "
                                 "model to be valid.")

        super().__init__(drug_list)

    def get_combined_effect(self,
                            dose_combination: List,
                            parameters: Union[np.array, List[List]] = None,
                            gradient: bool = False):
        """
        Returns the effect of a dose combination predicted by the
        combination model.

        The combined effect is given for monotone increasing drugs by

            f_{Bliss}(C) =   1 - \prod_i (1-f_i(c_i))

        (this is derived as 1 - "the probability, that every drug does
        not bind.")

        or for monotone decreasing drugs via

            f_{Bliss}(C) = \prod_i f_i(c_i)

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
        responses = np.nan * np.empty_like(dose_combination)
        if not gradient:

            for i, dose in enumerate(dose_combination):
                responses[i] = \
                    self.get_single_drug_response(idx=i,
                                                  dose=dose,
                                                  parameters=parameters,
                                                  gradient=gradient)

            if self.drug_list[0].monotone_increasing:
                return 1 - np.prod(1 - np.array(responses))
            else:
                return np.prod(responses)
        else:  # return gradient as second argument...

            gradients = list()
            # compute + append responses/gradients
            for i, dose in enumerate(dose_combination):
                responses[i], gradients_i = \
                    self.get_single_drug_response(idx=i,
                                                  dose=dose,
                                                  parameters=parameters,
                                                  gradient=gradient)

                gradients.append(gradients_i)

            # compute the response and the reduced product
            if self.drug_list[0].monotone_increasing:
                reduced_response_product = _index_reduced_product(1-responses)
                response = 1 - np.prod(1 - np.array(responses))
            else:
                reduced_response_product = _index_reduced_product(responses)
                response = np.prod(responses)

            # compute the gradient and return
            for i in range(self.n_drugs):
                    gradients[i] = reduced_response_product[i]*gradients[i]

            return response, np.array(gradients).flatten()


def _index_reduced_product(x: np.ndarray):
    """
    Returns a vector, whose i-th entry is the product of all elements of x
    except x[i].
    """
    res = np.nan*np.empty_like(x)
    for i in range(len(x)):
        res[i] = np.prod(np.append(x[:i], x[i+1:]))
    return res
