import numpy as np


class Aggregator:
    """
    Aggregator class for handling model parameter or gradient aggregation in Federated Learning.
    """

    def __init__(self, method="mean"):
        """
        Initialize the aggregator with a specific aggregation method.
        :param method: The aggregation method. Options: "mean", "weighted", "secure".
        """
        self.method = method

    def aggregate(self, gradients_list, weights_list=None):
        """
        Perform aggregation based on the chosen method.
        :param gradients_list: List of gradients from worker nodes (numpy arrays).
        :param weights_list: List of weights (for weighted aggregation).
        :return: Aggregated gradient or parameter update.
        """
        if self.method == "mean":
            return self._mean_aggregate(gradients_list)
        elif self.method == "weighted":
            if weights_list is None:
                raise ValueError("Weights list must be provided for weighted aggregation.")
            return self._weighted_aggregate(gradients_list, weights_list)
        elif self.method == "secure":
            return self._secure_aggregate(gradients_list)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")

    def _mean_aggregate(self, gradients_list):
        """
        Perform simple mean aggregation.
        """
        return np.mean(gradients_list, axis=0)

    def _weighted_aggregate(self, gradients_list, weights_list):
        """
        Perform weighted aggregation.
        :param gradients_list: List of gradients from worker nodes (numpy arrays).
        :param weights_list: List of weights (for each gradient).
        """
        total_weight = sum(weights_list)
        weighted_sum = sum(g * w for g, w in zip(gradients_list, weights_list))
        return weighted_sum / total_weight

    def _secure_aggregate(self, gradients_list):
        """
        Placeholder for secure aggregation (to be implemented).
        """
        # TODO: Implement secure aggregation using homomorphic encryption or secret sharing.
        raise NotImplementedError("Secure aggregation is not yet implemented.")

    def initialize_global_weights(self, gradient_shape):
        """
        Initialize global model weights as zeros (same shape as gradients).
        :param gradient_shape: Shape of the gradient array.
        :return: Initialized global weights.
        """
        return np.zeros(gradient_shape)

