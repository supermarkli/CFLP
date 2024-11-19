import unittest
import grpc
import numpy as np
from concurrent import futures
import threading
import time

# Import the necessary modules
from proto import federated_pb2, federated_pb2_grpc
from master_node.grpc_server import FederatedLearningService
from worker_node.worker_node import WorkerNode

# Mock Worker Node class
class MockWorkerNode(WorkerNode):
    def __init__(self, data, target_column, master_address="localhost:50051"):
        """
        Initialize a Mock Worker Node with in-memory data.
        """
        import pandas as pd
        from io import StringIO

        # Convert in-memory data to a DataFrame
        csv_data = StringIO(data)
        self.local_data = pd.read_csv(csv_data)
        super().__init__(data_path=None, target_column=target_column, master_address=master_address)

        # Overwrite local_model to use in-memory data
        self.local_model.data = self.local_data.drop(columns=[target_column])
        self.local_model.target = self.local_data[target_column]


class TestIntegration(unittest.TestCase):
    """
    Integration test suite for Federated Learning platform.
    """
    @classmethod
    def setUpClass(cls):
        """
        Start the Master Node gRPC server for testing.
        """
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        cls.federated_service = FederatedLearningService()
        federated_pb2_grpc.add_FederatedLearningServicer_to_server(cls.federated_service, cls.server)
        cls.server.add_insecure_port('[::]:50051')

        # Run the server in a separate thread
        cls.server_thread = threading.Thread(target=cls.server.start, daemon=True)
        cls.server_thread.start()
        print("Master Node gRPC server started for integration test.")

        # Give the server some time to start
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        """
        Stop the Master Node gRPC server after testing.
        """
        cls.server.stop(0)
        print("Master Node gRPC server stopped after integration test.")

    def setUp(self):
        """
        Set up Mock Worker Nodes for testing.
        """
        # Define in-memory datasets for two mock Worker Nodes
        self.worker1_data = """feature1,feature2,target
        1.0,2.0,0
        1.5,2.5,1
        2.0,3.0,0
        """

        self.worker2_data = """feature1,feature2,target
        2.0,3.5,1
        2.5,4.0,0
        3.0,4.5,1
        """

        # Initialize Mock Worker Nodes
        self.worker1 = MockWorkerNode(self.worker1_data, target_column="target")
        self.worker2 = MockWorkerNode(self.worker2_data, target_column="target")

    def test_federated_learning_round(self):
        """
        Test a single round of federated learning with two Worker Nodes.
        """
        # Perform gradient computation and communication with Master Node
        self.worker1.compute_and_send_gradient()
        self.worker2.compute_and_send_gradient()

        # Verify that global weights are updated correctly
        global_weights = self.federated_service.global_weights
        self.assertIsNotNone(global_weights, "Global weights should not be None after a learning round.")
        self.assertTrue(np.all(global_weights != 0), "Global weights should be updated after receiving gradients.")

    def test_multiple_rounds(self):
        """
        Test multiple rounds of federated learning.
        """
        num_rounds = 3
        for _ in range(num_rounds):
            self.worker1.compute_and_send_gradient()
            self.worker2.compute_and_send_gradient()

        # Verify final global weights after multiple rounds
        global_weights = self.federated_service.global_weights
        self.assertIsNotNone(global_weights, "Global weights should not be None after multiple rounds.")
        self.assertTrue(np.all(global_weights != 0), "Global weights should be updated after multiple rounds.")
        print(f"Final global weights after {num_rounds} rounds: {global_weights}")


if __name__ == "__main__":
    unittest.main()
