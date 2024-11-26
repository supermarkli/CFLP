import numpy as np
import logging

# 设置日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Aggregator:
    """
    Aggregator class for handling model parameter or gradient aggregation in Federated Learning.
    Supports various aggregation methods including mean, weighted, and secure aggregation.
    """

    def __init__(self, method="mean"):
        """
        Initialize the aggregator with a specific aggregation method.
        :param method: The aggregation method. Options: "mean", "weighted", "secure", "min", "max", "quantile".
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
            if weights_list is None or len(weights_list) != len(gradients_list):
                raise ValueError("The length of weights_list must match the length of gradients_list.")
            return self._weighted_aggregate(gradients_list, weights_list)
        elif self.method == "secure":
            return self._secure_aggregate(gradients_list)
        elif self.method == "min":
            return self._min_aggregate(gradients_list)
        elif self.method == "max":
            return self._max_aggregate(gradients_list)
        elif self.method == "quantile":
            return self._quantile_aggregate(gradients_list)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")

    def _mean_aggregate(self, gradients_list):
        """
        Perform simple mean aggregation of gradients.
        """
        logger.debug(f"Performing mean aggregation with {len(gradients_list)} gradients.")
        aggregated_gradient = np.mean(gradients_list, axis=0)
        logger.debug(f"Aggregated gradient: {aggregated_gradient}")
        return aggregated_gradient

    def _weighted_aggregate(self, gradients_list, weights_list):
        """
        Perform weighted aggregation of gradients.
        :param gradients_list: List of gradients from worker nodes (numpy arrays).
        :param weights_list: List of weights for each gradient.
        :return: Aggregated weighted gradient.
        """
        logger.debug(f"Performing weighted aggregation with {len(gradients_list)} gradients and {len(weights_list)} weights.")
        total_weight = sum(weights_list)
        if total_weight == 0:
            raise ValueError("The total weight must be greater than 0.")
        weighted_sum = sum(g * w for g, w in zip(gradients_list, weights_list))
        result = weighted_sum / total_weight
        logger.debug(f"Aggregated weighted gradient: {result}")
        return result

    def _secure_aggregate(self, gradients_list):
        """
        Placeholder for secure aggregation (to be implemented).
        """
        logger.warning("Secure aggregation has not been implemented yet. This may be a work in progress.")
        raise NotImplementedError("Secure aggregation is not yet implemented. Please refer to documentation.")

    def _min_aggregate(self, gradients_list):
        """
        Perform aggregation by taking the minimum value for each parameter across all gradients.
        """
        logger.debug(f"Performing min aggregation with {len(gradients_list)} gradients.")
        result = np.min(gradients_list, axis=0)
        logger.debug(f"Aggregated min gradient: {result}")
        return result

    def _max_aggregate(self, gradients_list):
        """
        Perform aggregation by taking the maximum value for each parameter across all gradients.
        """
        logger.debug(f"Performing max aggregation with {len(gradients_list)} gradients.")
        result = np.max(gradients_list, axis=0)
        logger.debug(f"Aggregated max gradient: {result}")
        return result

    def _quantile_aggregate(self, gradients_list, quantile=0.5):
        """
        Perform aggregation using a quantile (default 0.5 for median).
        :param quantile: The quantile value (e.g., 0.5 for median, 0.9 for 90th percentile).
        """
        logger.debug(f"Performing quantile aggregation with {len(gradients_list)} gradients at quantile {quantile}.")
        result = np.quantile(gradients_list, quantile, axis=0)
        logger.debug(f"Aggregated quantile gradient: {result}")
        return result

    def initialize_global_weights(self, gradient_shape):
        """
        Initialize global model weights as zeros (same shape as gradients).
        :param gradient_shape: Shape of the gradient array.
        :return: Initialized global weights (numpy array).
        """
        logger.debug(f"Initializing global weights with shape {gradient_shape}.")
        return np.zeros(gradient_shape)
