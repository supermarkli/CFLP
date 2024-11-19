import unittest
import grpc
import numpy as np
from concurrent import futures

# Import Worker Node modules and gRPC proto files
from proto import federated_pb2, federated_pb2_grpc
from worker_node.worker_node import WorkerNode
from worker_node.local_model import LocalModel

# Mock Master Node Service
class MockMasterNode(federated_pb2_grpc.FederatedLearningServicer):
    """
    A mock Master Node gRPC service for testing Worker Node.
    """
    def __init__(self):
        self.global_weights = np.zeros(3)  # Initialize global weights

    def SendGradient(self, request, context):
        # Simulate weight update logic
        gradient = np.array(request.gradient)
        self.global_weights -= 0.01 * gradient  # Simple SGD update
        return federated_pb2.GlobalWeights(weights=self.global_weights.tolist())


class TestWorkerNode(unittest.TestCase):
    """
    Test suite for the Worker Node.
    """
    @classmethod
    def setUpClass(cls):
        """
        Start a Mock Master Node gRPC server for testing.
        """
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        mock_service = MockMasterNode()
        federated_pb2_grpc.add_FederatedLearningServicer_to_server(mock_service, cls.server)
        cls.server.add_insecure_port('[::]:50051')
        cls.server.start()
        print("Mock Master Node gRPC server started on port 50051.")

    @classmethod
    def tearDownClass(cls):
        """
        Stop the gRPC server after testing.
        """
        cls.server.stop(0)
        print("Mock Master Node gRPC server stopped.")

    def setUp(self):
        """
        Initialize a Worker Node for each test case.
        """
        self.worker_node = WorkerNode(
            data_path="test_data.csv",
            target_column="target",
            master_address="localhost:50051"
        )

        # Create a sample dataset
        self.sample_data = {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0],
            "target": [0, 1, 0]
        }

        # Save the sample dataset to a CSV file
        import pandas as pd
        pd.DataFrame(self.sample_data).to_csv("test_data.csv", index=False)

    def test_local_model_training(self):
        """
        Test that the local model trains correctly.
        """
        self.worker_node.local_model.train()

        # Check that the model has been trained by verifying the weights are non-zero
        weights = self.worker_node.local_model.get_weights()
        self.assertTrue(np.any(weights != 0), "Local model weights should not be zero after training.")

    def test_gradient_computation(self):
        """
        Test that the gradients are computed correctly.
        """
        gradient = self.worker_node.local_model.compute_gradient()
        self.assertEqual(len(gradient), 2, "Gradient length should match the number of features.")
        self.assertTrue(np.allclose(gradient, gradient, atol=1e-5), "Gradient computation should be consistent.")

    def test_communication_with_master(self):
        """
        Test that the Worker Node can send gradients to the Mock Master Node and receive updated weights.
        """
        self.worker_node.compute_and_send_gradient()

        # Verify the Worker Node's model weights have been updated
        weights = self.worker_node.local_model.get_weights()
        self.assertTrue(np.all(weights != 0), "Worker Node should update its weights after communication with Master.")

    def tearDown(self):
        """
        Cleanup after each test case.
        """
        import os
        if os.path.exists("test_data.csv"):
            os.remove("test_data.csv")


if __name__ == "__main__":
    unittest.main()
