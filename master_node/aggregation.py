import numpy as np
import logging
from phe import paillier  # Using Paillier homomorphic encryption library

# Setting up logger
logging.basicConfig(
    level=logging.DEBUG,  # Set log level to DEBUG to ensure all logs are output
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.StreamHandler(),  # Output logs to console
        logging.FileHandler("federated_learning.log")  # Write logs to file
    ]
)

logger = logging.getLogger(__name__)

class Aggregator:
    """
    Aggregator class for handling secure aggregation in Federated Learning using various techniques.
    Supports Paillier homomorphic encryption.
    """

    def __init__(self, method="paillier"):
        """
        Initialize the aggregator with a specific security method.
        :param method: The security aggregation method. Options: "paillier".
        """
        self.method = method
        
        if self.method == "paillier":
            # Generate Paillier public and private keys
            self.public_key, self.private_key = paillier.generate_paillier_keypair()
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def aggregate(self, gradients_list):
        """
        Perform secure aggregation using the chosen method.
        :param gradients_list: List of gradients from worker nodes (in plaintext).
        :return: Aggregated result (encrypted).
        """
        if self.method == "paillier":
            return self._paillier_aggregate(gradients_list)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")

    def _paillier_aggregate(self, gradients_list):
        """
        Perform aggregation using Paillier homomorphic encryption.
        """
        logger.debug("Starting Paillier homomorphic encryption aggregation...")

        # Initialize encrypted sum to zero
        encrypted_sum = self.public_key.encrypt(0)

        # Encrypt each gradient and sum them homomorphically
        for gradient in gradients_list:
            # Ensure gradient is a flat list of floats (not numpy ndarray)
            if isinstance(gradient, np.ndarray):
                gradient = gradient.tolist()  # Convert numpy array to list of floats
            if not isinstance(gradient, list):
                gradient = [float(gradient)]  # Ensure it's a list of floats

            logger.debug(f"Encrypting gradient: {gradient}")
            
            # Encrypt each element in the gradient list
            for g in gradient:
                encrypted_gradient = self.public_key.encrypt(g)  # Encrypt the individual value
                encrypted_sum += encrypted_gradient

        logger.debug(f"Encrypted aggregated result: {encrypted_sum}")
        return encrypted_sum

    def decrypt(self, encrypted_data):
        """
        Decrypt the aggregated encrypted result.
        :param encrypted_data: The aggregated encrypted data.
        :return: Decrypted gradient (after aggregation).
        """
        if self.method == "paillier":
            decrypted_result = self.private_key.decrypt(encrypted_data)
            logger.debug(f"Decrypted aggregated result: {decrypted_result}")
            return decrypted_result
        else:
            raise ValueError("Decryption is only supported for Paillier encryption.")

    def initialize_global_weights(self, gradient_shape):
        """
        Initialize global model weights as zeros (same shape as gradients).
        :param gradient_shape: Shape of the gradient array.
        :return: Initialized global weights (numpy array).
        """
        logger.debug(f"Initializing global weights with shape {gradient_shape}.")
        return np.zeros(gradient_shape)