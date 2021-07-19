"""
Implements the Highest Single Agent Combination model.
"""
import numpy as np
from typing import List, Union

from .CombinationModelBase import CombinationModelBase
from ..DoseResponseModels import DoseResponseModelBase


class HSACombinationModel(CombinationModelBase):
    """
    Implements the Highest Single Agent Combination model, defined by

        f_{combination}(c) = max_i(f_i(c_i))

    for monotone increasing drugs and for monotone decreasing drugs via

        f_{combination}(c) = min_i(f_i(c_i)).
    """

    def __init__(self,
                 drug_list: List[DoseResponseModelBase]):
        """
        Constructor.
        """
        super().__init__(drug_list)

    def get_combined_effect(self,
                            dose_combination: List,
                            parameters: Union[np.array, List[List]] = None,
                            gradient: bool = False):
        """Computes the combined effect."""

        responses = np.nan * np.empty_like(dose_combination)
        if not gradient:

            for i, dose in enumerate(dose_combination):
                responses[i] = \
                    self.get_single_drug_response(idx=i,
                                                  dose=dose,
                                                  parameters=parameters,
                                                  gradient=gradient)

            if self.drug_list[0].monotone_increasing:
                return np.max(responses)
            else:
                return np.min(responses)
        else:  # return gradient as second argument...

            gradients = np.nan * np.empty((0, ))
            # compute + append responses/gradients
            for i, dose in enumerate(dose_combination):
                responses[i], gradients_i = \
                    self.get_single_drug_response(idx=i,
                                                  dose=dose,
                                                  parameters=parameters,
                                                  gradient=gradient)

                gradients = np.append(gradients, gradients_i)

            # find the min/max that should be returned
            if self.drug_list[0].monotone_increasing:
                idx = np.argmax(responses)
            else:
                idx = np.argmin(responses)

            i_min, i_max = self.get_index_range(idx)

            # The gradient is zero, except for the one drug,
            # that is the highest single agent...
            hsa_gradient = np.zeros_like(parameters)
            hsa_gradient[i_min:i_max] = gradients[i_min:i_max]

            return responses[idx], hsa_gradient
