import unittest
import grpc
from concurrent import futures
import numpy as np
import logging

# Import the generated gRPC modules and the server implementation
from proto import federated_pb2, federated_pb2_grpc
from master_node.master_node import FederatedLearningService


# Set up logging to capture detailed information
logging.basicConfig(
    level=logging.DEBUG,  # Log everything from DEBUG level up
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp, log level, and message
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("federated_learning.log")  # Log to a file
    ]
)
logger = logging.getLogger(__name__)  # Get a logger instance


class MockWorkerNode:
    """
    A mock worker node client for testing the Master Node gRPC server.
    """
    def __init__(self, server_address="localhost:50051"):
        self.server_address = server_address

    def send_gradient(self, gradient):
        """
        Send a gradient to the Master Node and receive updated global weights.
        :param gradient: Gradient array to send.
        :return: Updated global weights received from the Master Node.
        """
        logger.debug(f"MockWorkerNode sending gradient: {gradient}")
        with grpc.insecure_channel(self.server_address) as channel:
            stub = federated_pb2_grpc.FederatedLearningStub(channel)
            request = federated_pb2.Gradient(gradient=gradient.tolist())
            response = stub.SendGradient(request)
            logger.debug(f"MockWorkerNode received updated global weights: {response.weights}")
            return response.weights


class TestMasterNode(unittest.TestCase):
    """
    Test suite for the Master Node.
    """
    @classmethod
    def setUpClass(cls):
        """
        Start the gRPC server for testing.
        """
        logger.info("Starting the gRPC server for testing...")
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Initialize FederatedLearningService with Paillier aggregation method
        federated_service = FederatedLearningService(aggregation_method="paillier", learning_rate=0.01)
        federated_pb2_grpc.add_FederatedLearningServicer_to_server(federated_service, cls.server)
        cls.server.add_insecure_port('[::]:50051')
        cls.server.start()
        logger.info("Test gRPC server started on port 50051.")

    @classmethod
    def tearDownClass(cls):
        """
        Stop the gRPC server after testing.
        """
        logger.info("Stopping the test gRPC server...")
        cls.server.stop(0)
        logger.info("Test gRPC server stopped.")

    def test_send_gradient(self):
        """
        Test sending a gradient from a Worker Node and receiving updated weights.
        """
        logger.info("Starting test_send_gradient...")
        mock_worker = MockWorkerNode()
        gradient = [0.1, 0.2, 0.3]
        updated_weights = mock_worker.send_gradient(gradient)

        # Check if the updated weights are calculated correctly
        # Assuming initial weights = 0 and learning rate = 0.01, so expected updated weights would be:
        expected_weights = [-0.001, -0.002, -0.003]  # gradient * -learning_rate
        logger.info(f"Expected weights: {expected_weights}")
        
        np.testing.assert_almost_equal(updated_weights, expected_weights, decimal=5)
        logger.info("test_send_gradient passed.")

    def test_multiple_gradients(self):
        """
        Test the aggregation of multiple gradients sent sequentially.
        """
        logger.info("Starting test_multiple_gradients...")
        mock_worker = MockWorkerNode()
        gradients = [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.3, 0.3, 0.2]
        ]

        # Send gradients sequentially and check the final weights
        logger.debug(f"Sending gradients: {gradients}")
        for gradient in gradients:
            mock_worker.send_gradient(gradient)

        # Calculate the expected weights
        total_gradient = np.array([0.1, 0.2, 0.3]) + np.array([0.2, 0.1, 0.4]) + np.array([0.3, 0.3, 0.2])
        expected_weights = -0.01 * total_gradient  # Assuming learning_rate = 0.01
        logger.info(f"Expected aggregated weights: {expected_weights}")

        # Send a dummy gradient to retrieve the final updated global weights
        updated_weights = mock_worker.send_gradient([0, 0, 0])

        # Compare the final weights with the expected values
        np.testing.assert_almost_equal(updated_weights, expected_weights, decimal=5)
        logger.info("test_multiple_gradients passed.")


if __name__ == "__main__":
    logger.info("Starting the test suite...")
    unittest.main()
