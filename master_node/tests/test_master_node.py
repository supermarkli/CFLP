import unittest
import grpc
from concurrent import futures
import numpy as np

# Import the generated gRPC modules and the server implementation
from proto import federated_pb2, federated_pb2_grpc
from master_node.grpc_server import FederatedLearningService


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
        with grpc.insecure_channel(self.server_address) as channel:
            stub = federated_pb2_grpc.FederatedLearningStub(channel)
            request = federated_pb2.Gradient(gradient=gradient.tolist())
            response = stub.SendGradient(request)
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
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        federated_service = FederatedLearningService()
        federated_pb2_grpc.add_FederatedLearningServicer_to_server(federated_service, cls.server)
        cls.server.add_insecure_port('[::]:50051')
        cls.server.start()
        print("Test gRPC server started on port 50051.")

    @classmethod
    def tearDownClass(cls):
        """
        Stop the gRPC server after testing.
        """
        cls.server.stop(0)
        print("Test gRPC server stopped.")

    def test_send_gradient(self):
        """
        Test sending a gradient from a Worker Node and receiving updated weights.
        """
        mock_worker = MockWorkerNode()
        gradient = [0.1, 0.2, 0.3]
        updated_weights = mock_worker.send_gradient(gradient)

        # Check if the updated weights are calculated correctly
        expected_weights = [-0.001, -0.002, -0.003]  # Assuming initial weights = 0 and learning rate = 0.01
        np.testing.assert_almost_equal(updated_weights, expected_weights, decimal=5)

    def test_multiple_gradients(self):
        """
        Test the aggregation of multiple gradients sent sequentially.
        """
        mock_worker = MockWorkerNode()
        gradients = [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.3, 0.3, 0.2]
        ]

        # Send gradients sequentially and check the final weights
        for gradient in gradients:
            mock_worker.send_gradient(gradient)

        # Calculate the expected weights
        total_gradient = np.array([0.1, 0.2, 0.3]) + np.array([0.2, 0.1, 0.4]) + np.array([0.3, 0.3, 0.2])
        expected_weights = -0.01 * total_gradient
        updated_weights = mock_worker.send_gradient([0, 0, 0])  # Dummy gradient to retrieve final weights

        np.testing.assert_almost_equal(updated_weights, expected_weights, decimal=5)


if __name__ == "__main__":
    unittest.main()
